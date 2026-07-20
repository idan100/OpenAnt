"""Tests for PoolAdapter — active round-robin load balancing — and
build_phase_registry wiring a phase's ``pool`` field."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utilities.llm import (
    PHASES,
    CompletionResult,
    ConfigError,
    ConfigFile,
    LLMConfig,
    LLMRateLimitError,
    Message,
    PhaseRef,
    PoolMember,
    ProviderConfig,
    TextBlock,
    ToolUseBlock,
    build_phase_registry,
    empty_config,
    with_provider,
)
from utilities.llm.providers.pool import PoolAdapter
import utilities.rate_limiter as rl


def _all_phases_ref(provider: str, model: str) -> dict[str, PhaseRef]:
    return {p: PhaseRef(provider=provider, model=model) for p in PHASES}


def _ok_result(text="ok") -> CompletionResult:
    return CompletionResult(content=[TextBlock(text)], input_tokens=1, output_tokens=1, stop_reason="end_turn")


class _FakeAdapter:
    supports_tools = True

    def __init__(self, name=None, *, api_key=None, base_url=None):
        self.name = name
        self.complete_calls: list[dict] = []
        self.validate_calls: list[str] = []
        self.fail_with: Exception | None = None

    def complete(self, *, model, **kwargs):
        self.complete_calls.append({"model": model, **kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        return _ok_result(f"from-{self.name}")

    def validate(self, model):
        self.validate_calls.append(model)


class _FakeNoToolAdapter(_FakeAdapter):
    supports_tools = False


@pytest.fixture(autouse=True)
def _reset_state():
    rl.reset_rate_limiter()
    rl.reset_rpm_pacers()
    yield
    rl.reset_rate_limiter()
    rl.reset_rpm_pacers()


# ---------------------------------------------------------------------------
# PoolAdapter
# ---------------------------------------------------------------------------


class TestPoolAdapter:
    def _pool(self, n=3):
        adapters = [_FakeAdapter(f"vendor{i}") for i in range(n)]
        candidates = [(a, f"model{i}", f"vendor{i}") for i, a in enumerate(adapters)]
        return PoolAdapter(candidates=candidates, phase="analyze"), adapters

    def test_requires_at_least_one_candidate(self):
        with pytest.raises(ValueError):
            PoolAdapter(candidates=[], phase="analyze")

    def test_round_robins_start_index_across_calls(self):
        pool, adapters = self._pool(3)
        used = []
        for _ in range(3):
            result = pool.complete(model="ignored", system=None, messages=[], max_tokens=8)
            used.append(result.content[0].text)
        # All three distinct candidates got used exactly once across
        # three calls (rotation, not always-the-first).
        assert set(used) == {"from-vendor0", "from-vendor1", "from-vendor2"}

    def test_ignores_caller_supplied_model_uses_own(self):
        pool, adapters = self._pool(1)
        pool.complete(model="caller-said-this", system=None, messages=[], max_tokens=8)
        assert adapters[0].complete_calls[0]["model"] == "model0"

    def test_complete_logs_which_candidate_was_dispatched(self, capsys):
        pool, adapters = self._pool(2)
        pool.complete(model="x", system=None, messages=[], max_tokens=8)
        err = capsys.readouterr().err
        assert "[pool:analyze] -> vendor0/model0" in err

    def test_complete_logs_rate_limited_candidate_before_moving_on(self, capsys):
        pool, adapters = self._pool(2)
        adapters[0].fail_with = LLMRateLimitError("busy", retry_after=5)
        pool._next_index = 0
        pool.complete(model="x", system=None, messages=[], max_tokens=8)
        err = capsys.readouterr().err
        assert "[pool:analyze] -> vendor0/model0" in err
        assert "vendor0/model0 rate-limited, trying next candidate" in err
        assert "[pool:analyze] -> vendor1/model1" in err

    def test_skips_candidate_in_backoff(self):
        pool, adapters = self._pool(2)
        rl.get_rate_limiter("vendor0").report_rate_limit(60.0)  # vendor0 now in backoff
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text == "from-vendor1"
        assert adapters[0].complete_calls == []  # never even tried

    def test_skips_candidate_with_no_rpm_slot(self):
        pool, adapters = self._pool(2)
        rl.configure_rpm_limit("vendor0", "model0", 1)
        rl.get_rpm_pacer("vendor0", "model0").wait_for_slot()  # consume the only slot
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text == "from-vendor1"
        assert adapters[0].complete_calls == []

    def test_falls_through_to_busy_candidate_if_all_busy(self, monkeypatch):
        import utilities.llm.providers.pool as pool_module

        monkeypatch.setattr(pool_module.time, "sleep", lambda _: None)
        pool, adapters = self._pool(2)
        rl.get_rate_limiter("vendor0").report_rate_limit(60.0)
        rl.get_rate_limiter("vendor1").report_rate_limit(60.0)
        # Both busy — still succeeds by trying one of them anyway
        # rather than refusing outright (after the central "everyone's
        # busy" wait exhausts its bounded rounds, per _MAX_CENTRAL_WAIT_ROUNDS).
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text in ("from-vendor0", "from-vendor1")

    def test_rate_limit_on_one_candidate_tries_the_next(self):
        pool, adapters = self._pool(2)
        adapters[0].fail_with = LLMRateLimitError("busy", retry_after=5)
        # Force rotation to start at vendor0 so we know it's tried first.
        pool._next_index = 0
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text == "from-vendor1"

    # -----------------------------------------------------------------
    # "Everyone busy" central wait: when every candidate is currently
    # rate-limited/out-of-slots, don't commit to trying them in fixed
    # rotation order (which just means blocking on whichever one the
    # rotation lands on first) — check who clears soonest and sleep
    # exactly that long, then re-evaluate from scratch.
    # -----------------------------------------------------------------

    def test_waits_for_the_soonest_candidate_not_rotation_order(self):
        import time as time_module

        pool, adapters = self._pool(2)
        # vendor0 has the LONGER backoff, vendor1 the shorter one.
        # Rotation starts at vendor0 (index 0) — the old behavior would
        # commit to trying vendor0 first regardless. Set _backoff_until
        # directly rather than via report_rate_limit(), which clamps to
        # max(given, the limiter's own 30s default) -- both of these
        # short durations would otherwise silently become 30s.
        now = time_module.monotonic()
        rl.get_rate_limiter("vendor0")._backoff_until = now + 2.0
        rl.get_rate_limiter("vendor1")._backoff_until = now + 0.1
        pool._next_index = 0

        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)

        # The pool waited for the SOONER candidate (vendor1, ~0.1s) and
        # got it, rather than blocking on vendor0's longer 2s backoff.
        assert result.content[0].text == "from-vendor1"

    def test_does_not_sleep_when_soonest_wait_exceeds_threshold(self, monkeypatch):
        import utilities.llm.providers.pool as pool_module

        slept = []
        monkeypatch.setattr(pool_module.time, "sleep", lambda s: slept.append(s))
        pool, adapters = self._pool(2)
        # Both candidates busy for well over _MAX_CENTRAL_WAIT_SECONDS —
        # must fall straight through to best-effort dispatch, no central wait.
        rl.get_rate_limiter("vendor0").report_rate_limit(120.0)
        rl.get_rate_limiter("vendor1").report_rate_limit(120.0)

        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)

        assert result.content[0].text in ("from-vendor0", "from-vendor1")
        assert slept == [], "must not sleep centrally when nothing clears within the threshold"

    def test_gives_up_after_bounded_rounds_if_backoff_never_actually_clears(self, monkeypatch):
        import utilities.llm.providers.pool as pool_module

        # Sleep is a no-op, so real time never actually advances --
        # backoff never clears no matter how many rounds run. Must
        # still terminate (not hang) and fall through to dispatch.
        monkeypatch.setattr(pool_module.time, "sleep", lambda _: None)
        pool, adapters = self._pool(2)
        rl.get_rate_limiter("vendor0").report_rate_limit(1.0)
        rl.get_rate_limiter("vendor1").report_rate_limit(1.0)

        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)

        assert result.content[0].text in ("from-vendor0", "from-vendor1")

    def test_raises_when_every_candidate_rate_limited(self):
        pool, adapters = self._pool(2)
        for a in adapters:
            a.fail_with = LLMRateLimitError("busy", retry_after=5)
        with pytest.raises(LLMRateLimitError):
            pool.complete(model="x", system=None, messages=[], max_tokens=8)

    # -----------------------------------------------------------------
    # Sticky-per-conversation: a multi-turn tool-calling loop must stay
    # on the SAME candidate across turns, not round-robin per call —
    # see the module docstring. A "continuation" call is detected by
    # the presence of an assistant-role message in `messages`.
    # -----------------------------------------------------------------

    _FIRST_TURN = [Message(role="user", content=[TextBlock("hi")])]

    def _second_turn(self, tool_name="search"):
        return self._FIRST_TURN + [
            Message(role="assistant", content=[ToolUseBlock(id="t1", name=tool_name, input={})]),
            Message(role="user", content=[TextBlock("tool result")]),
        ]

    def test_continuation_call_stays_on_pinned_candidate(self):
        pool, adapters = self._pool(3)
        first = pool.complete(model="x", system=None, messages=self._FIRST_TURN, max_tokens=8)
        pinned_vendor = first.content[0].text  # e.g. "from-vendor0"

        # Without the fix, the next call would round-robin to vendor1 —
        # a different provider being asked to replay vendor0's tool call.
        second = pool.complete(model="x", system=None, messages=self._second_turn(), max_tokens=8)
        assert second.content[0].text == pinned_vendor

        third = pool.complete(model="x", system=None, messages=self._second_turn(), max_tokens=8)
        assert third.content[0].text == pinned_vendor

    def test_fresh_conversation_after_a_completed_one_advances_rotation(self):
        pool, adapters = self._pool(3)
        pool._next_index = 0
        first_conv = pool.complete(model="x", system=None, messages=self._FIRST_TURN, max_tokens=8)
        pool.complete(model="x", system=None, messages=self._second_turn(), max_tokens=8)  # continuation, stays pinned

        # A NEW conversation (fresh, no assistant turn yet) must advance
        # past whatever candidate served the previous conversation.
        second_conv = pool.complete(model="x", system=None, messages=self._FIRST_TURN, max_tokens=8)
        assert second_conv.content[0].text != first_conv.content[0].text

    def test_continuation_ignores_which_candidate_rotation_would_currently_point_at(self):
        pool, adapters = self._pool(3)
        pool._next_index = 0
        first = pool.complete(model="x", system=None, messages=self._FIRST_TURN, max_tokens=8)
        pinned_vendor = first.content[0].text
        # Simulate other conversations having advanced the shared rotation
        # index in between (e.g. sibling worker threads on other units).
        pool._next_index = 2
        second = pool.complete(model="x", system=None, messages=self._second_turn(), max_tokens=8)
        assert second.content[0].text == pinned_vendor

    def test_sticky_pin_is_per_thread(self):
        import threading

        pool, adapters = self._pool(3)
        pool._next_index = 0
        results = {}

        def worker(key):
            first = pool.complete(model="x", system=None, messages=self._FIRST_TURN, max_tokens=8)
            second = pool.complete(model="x", system=None, messages=self._second_turn(), max_tokens=8)
            results[key] = (first.content[0].text, second.content[0].text)

        t1 = threading.Thread(target=worker, args=("a",))
        t1.start()
        t1.join()
        t2 = threading.Thread(target=worker, args=("b",))
        t2.start()
        t2.join()

        # Each thread's own continuation call stayed pinned to ITS OWN
        # first-turn candidate, independent of the other thread.
        assert results["a"][0] == results["a"][1]
        assert results["b"][0] == results["b"][1]

    # -----------------------------------------------------------------
    # Context-window routing: a candidate with a known per-request token
    # ceiling (e.g. GitHub Models' 8k-token "Low" tier) is skipped in
    # favor of a candidate without one, rather than 413ing.
    # -----------------------------------------------------------------

    def test_skips_candidate_whose_known_ceiling_the_request_exceeds(self, monkeypatch):
        import utilities.token_estimate as te

        pool, adapters = self._pool(2)
        # vendor0/model0 has a tiny known ceiling; vendor1/model1 has none.
        monkeypatch.setitem(te.KNOWN_MAX_REQUEST_TOKENS, ("vendor0", "model0"), 10)
        big_text = "x" * 1000  # ~250 estimated tokens, well over the 10-token ceiling
        pool._next_index = 0
        result = pool.complete(
            model="ignored", system=None,
            messages=[Message(role="user", content=[TextBlock(big_text)])],
            max_tokens=8,
        )
        assert result.content[0].text == "from-vendor1"
        assert adapters[0].complete_calls == [], "the oversized candidate must never even be tried"

    def test_uses_candidate_with_known_ceiling_when_request_fits(self, monkeypatch):
        import utilities.token_estimate as te

        pool, adapters = self._pool(2)
        monkeypatch.setitem(te.KNOWN_MAX_REQUEST_TOKENS, ("vendor0", "model0"), 10_000)
        pool._next_index = 0
        result = pool.complete(
            model="ignored", system=None,
            messages=[Message(role="user", content=[TextBlock("small request")])],
            max_tokens=8,
        )
        assert result.content[0].text == "from-vendor0"

    def test_tries_oversized_candidate_as_absolute_last_resort(self, monkeypatch):
        """If EVERY candidate is oversized (or busy), still try one rather
        than erroring outright — matches the existing "busy" fallthrough
        philosophy (some chance beats none)."""
        import utilities.token_estimate as te

        pool, adapters = self._pool(1)
        monkeypatch.setitem(te.KNOWN_MAX_REQUEST_TOKENS, ("vendor0", "model0"), 1)
        result = pool.complete(
            model="ignored", system=None,
            messages=[Message(role="user", content=[TextBlock("x" * 1000)])],
            max_tokens=8,
        )
        assert result.content[0].text == "from-vendor0"

    def test_no_known_ceiling_means_no_size_based_skipping(self):
        # No entry in KNOWN_MAX_REQUEST_TOKENS for either candidate ->
        # behaves exactly like the plain round-robin tests, regardless
        # of request size.
        pool, adapters = self._pool(2)
        pool._next_index = 0
        result = pool.complete(
            model="ignored", system=None,
            messages=[Message(role="user", content=[TextBlock("x" * 100_000)])],
            max_tokens=8,
        )
        assert result.content[0].text == "from-vendor0"

    def test_supports_tools_requires_all_members(self):
        good = _FakeAdapter("a")
        bad = _FakeNoToolAdapter("b")
        pool = PoolAdapter(candidates=[(good, "m", "a"), (bad, "m", "b")], phase="verify")
        assert pool.supports_tools is False

    def test_name_is_joined_provider_list(self):
        pool, _ = self._pool(2)
        assert pool.name == "vendor0+vendor1"

    def test_pricing_looks_up_whichever_candidate_has_the_model(self):
        a = _FakeAdapter("a")
        a.pricing = {"model-a": {"input": 1.0, "output": 2.0}}
        b = _FakeAdapter("b")
        b.pricing = {"model-b": {"input": 3.0, "output": 4.0}}
        pool = PoolAdapter(candidates=[(a, "model-a", "a"), (b, "model-b", "b")], phase="analyze")
        assert pool.pricing.get("model-a") == {"input": 1.0, "output": 2.0}
        assert pool.pricing.get("model-b") == {"input": 3.0, "output": 4.0}
        assert pool.pricing.get("unknown-model") is None

    def test_pricing_respects_a_candidate_with_custom_get_override(self):
        """The whole point of the fix: a candidate whose .pricing is a
        dict SUBCLASS with custom .get() behavior (like claude_sub's
        _ZeroCostPricing, which reports $0 for ANY key via an override
        rather than real stored items) must not be silently dropped by
        an eager dict.update() merge — see the PoolAdapter.pricing
        docstring."""
        class _AlwaysZeroPricing(dict):
            def get(self, key, default=None):
                return {"input": 0.0, "output": 0.0}

        zero_cost = _FakeAdapter("claude_sub")
        zero_cost.pricing = _AlwaysZeroPricing()
        other = _FakeAdapter("b")
        other.pricing = {"model-b": {"input": 3.0, "output": 4.0}}
        pool = PoolAdapter(candidates=[(zero_cost, "opus", "claude_sub"), (other, "model-b", "b")], phase="analyze")
        assert pool.pricing.get("opus") == {"input": 0.0, "output": 0.0}

    def test_validate_succeeds_if_at_least_one_member_ok(self):
        pool, adapters = self._pool(2)
        from utilities.llm import LLMAuthError
        adapters[0].validate = lambda model: (_ for _ in ()).throw(LLMAuthError("bad key"))
        pool.validate("ignored")  # must not raise
        assert adapters[1].validate_calls == ["model1"]

    def test_validate_raises_if_every_member_fails(self):
        pool, adapters = self._pool(2)
        from utilities.llm import LLMAuthError
        for a in adapters:
            a.validate = lambda model: (_ for _ in ()).throw(LLMAuthError("bad key"))
        with pytest.raises(LLMAuthError):
            pool.validate("ignored")


# ---------------------------------------------------------------------------
# build_phase_registry wiring a phase's `pool`
# ---------------------------------------------------------------------------


class TestBuildPhaseRegistryPool:
    def _cf(self) -> ConfigFile:
        cf = with_provider(empty_config(), ProviderConfig(name="anthropic", type="anthropic", api_key="sk"))
        cf = with_provider(cf, ProviderConfig(name="google", type="google", api_key="AIza"))
        cf = with_provider(cf, ProviderConfig(name="groq", type="openai", api_key="gsk", base_url="https://api.groq.com/openai/v1"))
        return cf

    def test_no_pool_means_plain_adapter(self):
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"))
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(self._cf(), primary)
        binding = registry.get("analyze")
        assert not isinstance(binding.adapter, PoolAdapter)

    def test_pool_field_produces_pool_adapter(self):
        phases = dict(_all_phases_ref("anthropic", "m"))
        phases["analyze"] = PhaseRef(
            provider="anthropic", model="m",
            pool=(PoolMember(provider="google", model="gm"), PoolMember(provider="groq", model="grm")),
        )
        cfg = LLMConfig(name="primary", phases=phases)
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(self._cf(), cfg)
        binding = registry.get("analyze")
        assert isinstance(binding.adapter, PoolAdapter)
        assert len(binding.adapter._candidates) == 3  # primary + 2 pool members
        # Untouched phase stays a plain adapter.
        assert not isinstance(registry.get("verify").adapter, PoolAdapter)

    def test_pool_members_share_adapter_instance_with_other_phases(self):
        # "google" used as a plain primary for app_context AND as a pool
        # member for analyze — must be the SAME adapter instance either way.
        phases = dict(_all_phases_ref("anthropic", "m"))
        phases["app_context"] = PhaseRef(provider="google", model="gm")
        phases["analyze"] = PhaseRef(
            provider="anthropic", model="m",
            pool=(PoolMember(provider="google", model="gm"),),
        )
        cfg = LLMConfig(name="primary", phases=phases)
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(self._cf(), cfg)
        app_context_adapter = registry.get("app_context").adapter
        pool_member_adapter = registry.get("analyze").adapter._candidates[1][0]
        assert app_context_adapter is pool_member_adapter

    def test_pool_member_missing_tool_support_rejected_for_tool_phase(self):
        phases = dict(_all_phases_ref("anthropic", "m"))
        phases["verify"] = PhaseRef(
            provider="anthropic", model="m",
            pool=(PoolMember(provider="groq", model="grm"),),
        )
        cfg = LLMConfig(name="primary", phases=phases)

        def _pick(type_name):
            return _FakeNoToolAdapter if type_name == "openai" else _FakeAdapter

        with patch("utilities.llm.registry.get_adapter_class", side_effect=_pick):
            with pytest.raises(ConfigError):
                build_phase_registry(self._cf(), cfg)

    def test_pool_and_fallback_compose(self):
        # Primary "analyze" pools anthropic+google; fallback config
        # points analyze at groq alone. Exhausting the WHOLE pool
        # should still be representable (structural check only here —
        # behavior is covered by TestPoolAdapter + TestFailoverAdapter
        # individually).
        primary_phases = dict(_all_phases_ref("anthropic", "m"))
        primary_phases["analyze"] = PhaseRef(
            provider="anthropic", model="m", pool=(PoolMember(provider="google", model="gm"),),
        )
        primary = LLMConfig(name="primary", phases=primary_phases, fallback="fb")
        fallback = LLMConfig(name="fb", phases=_all_phases_ref("groq", "grm"))
        cf = self._cf()
        cf.llm_configs["fb"] = fallback

        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(cf, primary)

        from utilities.llm.providers.failover import FailoverAdapter
        binding = registry.get("analyze")
        assert isinstance(binding.adapter, FailoverAdapter)
        assert isinstance(binding.adapter._primary, PoolAdapter)
