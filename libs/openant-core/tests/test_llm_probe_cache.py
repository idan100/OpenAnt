"""Tests for the (adapter, model) probe cache and its use in
PhaseRegistry.validate().

Cache path isolation (never touch the real
~/.config/openant/probe_cache.json) is handled globally by the
autouse ``_isolate_probe_cache`` fixture in ``tests/conftest.py`` —
every test in this suite already gets it for free.
"""

from __future__ import annotations

from unittest.mock import patch

from utilities.llm import probe_cache


class TestProbeCache:
    def test_not_validated_when_cache_empty(self):
        assert not probe_cache.was_recently_validated("google", "gemini-x")

    def test_recently_validated_is_true_within_ttl(self):
        probe_cache.mark_validated("google", "gemini-x")
        assert probe_cache.was_recently_validated("google", "gemini-x")

    def test_different_model_not_covered(self):
        probe_cache.mark_validated("google", "gemini-x")
        assert not probe_cache.was_recently_validated("google", "gemini-y")

    def test_expired_entry_is_false(self, monkeypatch):
        probe_cache.mark_validated("google", "gemini-x")
        real_time = probe_cache.time.time
        monkeypatch.setattr(probe_cache.time, "time", lambda: real_time() + 3600)
        assert not probe_cache.was_recently_validated("google", "gemini-x")

    def test_corrupt_cache_file_treated_as_empty(self):
        probe_cache._cache_path().parent.mkdir(parents=True, exist_ok=True)
        probe_cache._cache_path().write_text("{not valid json", encoding="utf-8")
        assert not probe_cache.was_recently_validated("google", "gemini-x")

    def test_persists_across_loads(self):
        probe_cache.mark_validated("anthropic", "claude-sonnet-5")
        # Simulate a fresh process re-reading the file from disk.
        assert probe_cache.was_recently_validated("anthropic", "claude-sonnet-5")


class TestValidateSkipsRecentlyProbed:
    def test_validate_skips_when_cached(self):
        from tests.test_llm_registry import _FakeAdapter, _all_phases_ref
        from utilities.llm import (
            LLMConfig,
            ProviderConfig,
            build_phase_registry,
            empty_config,
            with_provider,
        )

        _FakeAdapter.instances = []
        cf = with_provider(empty_config(), ProviderConfig(name="anthropic", type="anthropic", api_key="sk"))
        cfg = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"))
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(cf, cfg)

        probe_cache.mark_validated("anthropic", "m")
        registry.validate()

        adapter = _FakeAdapter.instances[0]
        assert adapter.validate_calls == []  # skipped — already cached
        _FakeAdapter.instances = []

    def test_validate_probes_and_caches_when_not_cached(self):
        from tests.test_llm_registry import _FakeAdapter, _all_phases_ref
        from utilities.llm import (
            LLMConfig,
            ProviderConfig,
            build_phase_registry,
            empty_config,
            with_provider,
        )

        _FakeAdapter.instances = []
        cf = with_provider(empty_config(), ProviderConfig(name="anthropic", type="anthropic", api_key="sk"))
        cfg = LLMConfig(name="primary", phases=_all_phases_ref("anthropic", "m"))
        with patch("utilities.llm.registry.get_adapter_class", return_value=_FakeAdapter):
            registry = build_phase_registry(cf, cfg)

        assert not probe_cache.was_recently_validated("anthropic", "m")
        registry.validate()

        adapter = _FakeAdapter.instances[0]
        assert adapter.validate_calls == ["m"]
        assert probe_cache.was_recently_validated("anthropic", "m")
        _FakeAdapter.instances = []
