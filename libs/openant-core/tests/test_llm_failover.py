"""Tests for provider failover — ExhaustionState, FailoverAdapter, and
build_phase_registry wiring a configured ``fallback`` llm-config."""

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
    ProviderConfig,
    TextBlock,
    build_phase_registry,
    empty_config,
    with_provider,
)
from utilities.llm.providers.failover import (
    ExhaustionState,
    FailoverAdapter,
    is_exhaustion_signal,
)


def _all_phases_ref(provider: str, model: str) -> dict[str, PhaseRef]:
    return {p: PhaseRef(provider=provider, model=model) for p in PHASES}


def _ok_result() -> CompletionResult:
    return CompletionResult(content=[TextBlock("ok")], input_tokens=1, output_tokens=1, stop_reason="end_turn")


class _FakeAdapter:
    name = "fake"
    supports_tools = True

    def __init__(self, *, api_key=None, base_url=None, name="fake"):
        self.api_key = api_key
        self.base_url = base_url
        self.name = name
        self.complete_calls: list[dict] = []
        self.validate_calls: list[str] = []
        self.fail_with: Exception | None = None

    def complete(self, *, model, **kwargs):
        self.complete_calls.append({"model": model, **kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        return _ok_result()

    def validate(self, model):
        self.validate_calls.append(model)


class _FakeNoToolAdapter(_FakeAdapter):
    supports_tools = False


# ---------------------------------------------------------------------------
# is_exhaustion_signal
# ---------------------------------------------------------------------------


class TestIsExhaustionSignal:
    def test_no_retry_after_is_exhaustion(self):
        assert is_exhaustion_signal(LLMRateLimitError("x", retry_after=None))

    def test_short_retry_after_is_not_exhaustion(self):
        assert not is_exhaustion_signal(LLMRateLimitError("x", retry_after=7.0))

    def test_long_retry_after_is_exhaustion(self):
        assert is_exhaustion_signal(LLMRateLimitError("x", retry_after=3600.0))


# ---------------------------------------------------------------------------
# ExhaustionState
# ---------------------------------------------------------------------------


class TestExhaustionState:
    def test_not_failed_over_initially(self):
        assert not ExhaustionState().is_failed_over()

    def test_requires_multiple_signals(self):
        state = ExhaustionState()
        assert state.report_signal() is False  # 1st signal: not yet
        assert not state.is_failed_over()
        assert state.report_signal() is True  # 2nd signal: flips, returns True once
        assert state.is_failed_over()

    def test_report_signal_returns_false_once_already_failed_over(self):
        state = ExhaustionState()
        state.report_signal()
        state.report_signal()
        assert state.is_failed_over()
        assert state.report_signal() is False  # already active, not a fresh flip


# ---------------------------------------------------------------------------
# FailoverAdapter
# ---------------------------------------------------------------------------


class TestFailoverAdapter:
    def _make(self):
        primary = _FakeAdapter(name="primary")
        fallback = _FakeAdapter(name="fallback")
        state = ExhaustionState()
        adapter = FailoverAdapter(
            primary=primary, primary_model="primary-model",
            fallback=fallback, fallback_model="fallback-model",
            phase="analyze", exhaustion_state=state,
        )
        return adapter, primary, fallback, state

    def test_normal_call_uses_primary(self):
        adapter, primary, fallback, _ = self._make()
        adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        assert primary.complete_calls == [{"model": "primary-model", "system": None, "messages": [], "max_tokens": 8}]
        assert fallback.complete_calls == []

    def test_complete_logs_which_side_was_dispatched(self, capsys):
        adapter, primary, fallback, state = self._make()
        adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        err = capsys.readouterr().err
        assert "[phase:analyze] -> primary/primary-model" in err

    def test_complete_logs_fallback_side_once_failed_over(self, capsys):
        adapter, primary, fallback, state = self._make()
        state.report_signal()
        state.report_signal()  # flips to failed-over
        capsys.readouterr()  # discard setup noise, if any
        adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        err = capsys.readouterr().err
        assert "[phase:analyze] -> fallback/fallback-model" in err

    def test_does_not_double_log_when_wrapping_a_pool_adapter(self, capsys):
        from utilities.llm.providers.pool import PoolAdapter

        pool_primary = PoolAdapter(
            candidates=[(_FakeAdapter(name="p1"), "m1", "p1")], phase="analyze",
        )
        fallback = _FakeAdapter(name="fb")
        adapter = FailoverAdapter(
            primary=pool_primary, primary_model="unused",
            fallback=fallback, fallback_model="fb-model",
            phase="analyze", exhaustion_state=ExhaustionState(),
        )
        adapter.complete(model="x", system=None, messages=[], max_tokens=8)
        err = capsys.readouterr().err
        # Only the pool's own dispatch line — no separate/duplicate
        # "[phase:analyze] -> ..." line from FailoverAdapter itself.
        assert "[pool:analyze] -> p1/m1" in err
        assert "[phase:analyze] ->" not in err

    def test_transient_rate_limit_propagates_without_failover(self):
        adapter, primary, fallback, state = self._make()
        primary.fail_with = LLMRateLimitError("slow down", retry_after=5.0)
        with pytest.raises(LLMRateLimitError):
            adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        assert not state.is_failed_over()
        assert fallback.complete_calls == []

    def test_exhaustion_signal_switches_to_fallback_after_threshold(self):
        adapter, primary, fallback, state = self._make()
        primary.fail_with = LLMRateLimitError("cap hit", retry_after=None)

        # 1st exhaustion-looking call: not enough signals yet, propagates.
        with pytest.raises(LLMRateLimitError):
            adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        assert not state.is_failed_over()

        # 2nd: threshold reached, transparently retried against fallback
        # using the FALLBACK's own model, not the caller-supplied one.
        result = adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        assert result.content[0].text == "ok"
        assert state.is_failed_over()
        assert fallback.complete_calls[-1]["model"] == "fallback-model"

        # 3rd call: already failed over, goes straight to fallback, no
        # more primary calls.
        adapter.complete(model="primary-model", system=None, messages=[], max_tokens=8)
        assert len(primary.complete_calls) == 2  # only the two exhaustion attempts above
        assert len(fallback.complete_calls) == 2

    def test_shared_state_across_two_phase_adapters(self):
        primary = _FakeAdapter(name="primary")
        fallback = _FakeAdapter(name="fallback")
        state = ExhaustionState()
        analyze = FailoverAdapter(primary=primary, primary_model="m", fallback=fallback, fallback_model="fb", phase="analyze", exhaustion_state=state)
        verify = FailoverAdapter(primary=primary, primary_model="m", fallback=fallback, fallback_model="fb", phase="verify", exhaustion_state=state)

        primary.fail_with = LLMRateLimitError("cap hit", retry_after=None)
        with pytest.raises(LLMRateLimitError):
            analyze.complete(model="m", system=None, messages=[], max_tokens=8)
        # verify's OWN adapter hasn't failed yet, but shares the state —
        # its next call should already see the (about to flip) state.
        verify.complete(model="m", system=None, messages=[], max_tokens=8)  # this is the 2nd signal -> flips
        assert state.is_failed_over()
        assert fallback.complete_calls  # verify's retry landed on fallback

    def test_supports_tools_requires_both_sides(self):
        primary = _FakeAdapter(name="primary")
        fallback = _FakeNoToolAdapter(name="fallback")
        adapter = FailoverAdapter(primary=primary, primary_model="m", fallback=fallback, fallback_model="fb", phase="verify", exhaustion_state=ExhaustionState())
        assert adapter.supports_tools is False

    def test_name_reflects_active_side(self):
        adapter, primary, fallback, state = self._make()
        assert adapter.name == "primary"
        state.report_signal()
        state.report_signal()
        assert adapter.name == "fallback"

    def test_validate_targets_primary_before_failover(self):
        adapter, primary, fallback, _ = self._make()
        adapter.validate("whatever")
        assert primary.validate_calls == ["primary-model"]
        assert fallback.validate_calls == []


# ---------------------------------------------------------------------------
# build_phase_registry wiring
# ---------------------------------------------------------------------------


class TestBuildPhaseRegistryFailover:
    def _cf(self) -> ConfigFile:
        cf = with_provider(empty_config(), ProviderConfig(name="anthropic", type="anthropic", api_key="sk"))
        cf = with_provider(cf, ProviderConfig(name="google", type="google", api_key="AIza"))
        return cf

    def test_no_fallback_means_plain_adapter(self):
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"))
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(self._cf(), primary)
        binding = registry.get("analyze")
        assert isinstance(binding.adapter, _FakeAdapter)  # not wrapped

    def test_fallback_wraps_every_phase_in_failover_adapter(self):
        fallback_cfg = LLMConfig(name="fb", phases=_all_phases_ref("google", "gm"))
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"), fallback="fb")
        cf = self._cf()
        cf.llm_configs["fb"] = fallback_cfg
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(cf, primary)
        for phase in PHASES:
            assert isinstance(registry.get(phase).adapter, FailoverAdapter)

    def test_unknown_fallback_name_raises(self):
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"), fallback="nonexistent")
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            with pytest.raises(ConfigError):
                build_phase_registry(self._cf(), primary)

    def test_validate_probes_fallback_too(self):
        fallback_cfg = LLMConfig(name="fb", phases=_all_phases_ref("google", "gm"))
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"), fallback="fb")
        cf = self._cf()
        cf.llm_configs["fb"] = fallback_cfg
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(cf, primary)
            registry.validate()
        # Primary probed once (all 7 phases share one provider+model).
        binding = registry.get("analyze")
        assert binding.adapter._primary.validate_calls == ["m"]
        assert binding.adapter._fallback.validate_calls == ["gm"]

    def test_fallback_missing_tool_support_raises_for_tool_phase(self):
        fallback_cfg = LLMConfig(name="fb", phases=_all_phases_ref("google", "gm"))
        primary = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"), fallback="fb")
        cf = self._cf()
        cf.llm_configs["fb"] = fallback_cfg

        def _pick(type_name):
            return _FakeNoToolAdapter if type_name == "google" else _FakeAdapter

        with patch("utilities.llm.registry.get_adapter_class", side_effect=_pick):
            with pytest.raises(ConfigError):
                build_phase_registry(cf, primary)
