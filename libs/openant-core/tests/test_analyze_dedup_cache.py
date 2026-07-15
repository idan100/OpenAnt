"""Stage 1 exact-duplicate response cache.

``analyze_unit`` builds a fully-rendered (system_prompt, prompt) pair per
unit and calls the model. Two units only produce the exact same rendered
text when every input that affects it — code, route, files_included,
security_classification, app_context — is identical, in which case the
model would see byte-identical input and there is nothing new to learn by
asking again. ``analyze_unit`` short-circuits that case and reuses the
cached response instead of spending another API call.

These tests stub the adapter layer — no network calls.
"""

from __future__ import annotations

import pytest

from experiment import analyze_unit, reset_analyze_cache
from utilities.llm import CompletionResult, PhaseBinding, TextBlock


class _CountingAdapter:
    name = "anthropic"
    supports_tools = False

    def __init__(self):
        self.calls = 0

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls += 1
        return CompletionResult(
            content=[TextBlock('{"finding": "safe", "reasoning": "x", "confidence": 0.9}')],
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )

    def validate(self, model):
        pass


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_analyze_cache()
    yield
    reset_analyze_cache()


def _binding(adapter):
    return PhaseBinding(phase="analyze", adapter=adapter, model="claude-test", provider_name="test")


def _unit(code: str, unit_id: str = "a.py:fn"):
    return {
        "id": unit_id,
        "unit_type": "function",
        "code": {"primary_code": code, "primary_origin": {}},
        "metadata": {"direct_calls": [], "direct_callers": []},
    }


class TestExactDuplicateCache:
    def test_identical_units_hit_the_adapter_once(self):
        adapter = _CountingAdapter()
        binding = _binding(adapter)

        analyze_unit(binding, _unit("def fn(): return 1", unit_id="a.py:fn"))
        analyze_unit(binding, _unit("def fn(): return 1", unit_id="b.py:fn"))

        assert adapter.calls == 1

    def test_different_code_hits_the_adapter_each_time(self):
        adapter = _CountingAdapter()
        binding = _binding(adapter)

        analyze_unit(binding, _unit("def fn(): return 1"))
        analyze_unit(binding, _unit("def fn(): return 2"))

        assert adapter.calls == 2

    def test_reset_clears_the_cache(self):
        adapter = _CountingAdapter()
        binding = _binding(adapter)

        analyze_unit(binding, _unit("def fn(): return 1"))
        reset_analyze_cache()
        analyze_unit(binding, _unit("def fn(): return 1"))

        assert adapter.calls == 2

    def test_cached_result_still_parses_and_normalizes(self):
        # A cache hit must go through the same parse/normalize path as a
        # fresh call — only the API round-trip is skipped.
        adapter = _CountingAdapter()
        binding = _binding(adapter)

        first = analyze_unit(binding, _unit("def fn(): return 1", unit_id="a.py:fn"))
        second = analyze_unit(binding, _unit("def fn(): return 1", unit_id="b.py:fn"))

        assert first["finding"] == "safe"
        assert second["finding"] == "safe"
        # route_key still reflects the SECOND unit's own identity, not a
        # stale copy of the first unit's result.
        assert second["route_key"] == "b.py:fn"
