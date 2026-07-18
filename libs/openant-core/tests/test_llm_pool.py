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
    PhaseRef,
    PoolMember,
    ProviderConfig,
    TextBlock,
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

    def test_falls_through_to_busy_candidate_if_all_busy(self):
        pool, adapters = self._pool(2)
        rl.get_rate_limiter("vendor0").report_rate_limit(60.0)
        rl.get_rate_limiter("vendor1").report_rate_limit(60.0)
        # Both busy — still succeeds by trying one of them anyway
        # rather than refusing outright.
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text in ("from-vendor0", "from-vendor1")

    def test_rate_limit_on_one_candidate_tries_the_next(self):
        pool, adapters = self._pool(2)
        adapters[0].fail_with = LLMRateLimitError("busy", retry_after=5)
        # Force rotation to start at vendor0 so we know it's tried first.
        pool._next_index = 0
        result = pool.complete(model="x", system=None, messages=[], max_tokens=8)
        assert result.content[0].text == "from-vendor1"

    def test_raises_when_every_candidate_rate_limited(self):
        pool, adapters = self._pool(2)
        for a in adapters:
            a.fail_with = LLMRateLimitError("busy", retry_after=5)
        with pytest.raises(LLMRateLimitError):
            pool.complete(model="x", system=None, messages=[], max_tokens=8)

    def test_supports_tools_requires_all_members(self):
        good = _FakeAdapter("a")
        bad = _FakeNoToolAdapter("b")
        pool = PoolAdapter(candidates=[(good, "m", "a"), (bad, "m", "b")], phase="verify")
        assert pool.supports_tools is False

    def test_name_is_joined_provider_list(self):
        pool, _ = self._pool(2)
        assert pool.name == "vendor0+vendor1"

    def test_pricing_merges_all_members(self):
        a = _FakeAdapter("a")
        a.pricing = {"model-a": {"input": 1.0, "output": 2.0}}
        b = _FakeAdapter("b")
        b.pricing = {"model-b": {"input": 3.0, "output": 4.0}}
        pool = PoolAdapter(candidates=[(a, "model-a", "a"), (b, "model-b", "b")], phase="analyze")
        assert pool.pricing == {
            "model-a": {"input": 1.0, "output": 2.0},
            "model-b": {"input": 3.0, "output": 4.0},
        }

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
