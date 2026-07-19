"""Tests for FindingVerifier's deterministic caller-guard pre-check
(_find_sibling_call_sites) and its injection into the verify prompt.

This targets the false-positive shape where Stage 1 flags a function in
isolation, but every REAL caller already validates the exact
precondition Stage 1 claims is missing. Finding those siblings is a
plain static search (RepositoryIndex.search_usages), computed once
up front and handed to the verifier as ready-made context — not
something left to the model to rediscover via its own tool calls.
"""

from __future__ import annotations

import pytest

from utilities.agentic_enhancer.repository_index import RepositoryIndex
from utilities.finding_verifier import FindingVerifier
from utilities.llm import LLMResponseError, PhaseBinding, TextBlock, ToolUseBlock
from utilities.llm.adapter import CompletionResult
from utilities.llm_client import reset_warning_state


@pytest.fixture(autouse=True)
def _reset():
    reset_warning_state()
    yield
    reset_warning_state()


def _index_with(functions: dict) -> RepositoryIndex:
    return RepositoryIndex({"functions": functions}, repo_path=None)


def _make_verifier(index: RepositoryIndex, adapter=None) -> FindingVerifier:
    adapter = adapter or _RecordingAdapter()
    binding = PhaseBinding(phase="verify", adapter=adapter, model="claude-x", provider_name="anthropic")
    return FindingVerifier(index=index, binding=binding), adapter


class _RecordingAdapter:
    """Records every messages/system it's called with; always finishes
    with a real, incomplete=False verdict on the first turn so tests
    that only care about PROMPT CONTENT don't need to model the loop."""

    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self):
        self.calls = []

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        return CompletionResult(
            content=[ToolUseBlock(
                id="finish-1", name="finish",
                input={"agree": True, "correct_finding": "safe", "explanation": "no issue"},
            )],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
        )


class TestFindSiblingCallSites:
    def test_no_function_id_returns_empty(self):
        verifier, _ = _make_verifier(_index_with({}))
        assert verifier._find_sibling_call_sites(None) == ""

    def test_no_index_data_returns_empty(self):
        verifier, _ = _make_verifier(_index_with({}))
        assert verifier._find_sibling_call_sites("some/file.c:target") == ""

    def test_no_siblings_found_returns_empty(self):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) { do_thing(n); }"},
        })
        verifier, _ = _make_verifier(index)
        assert verifier._find_sibling_call_sites("some/file.c:target") == ""

    def test_finds_and_formats_a_sibling_caller(self):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) { do_thing(n); }"},
            "some/other.c:caller": {
                "name": "caller",
                "code": "void caller(int n) {\n  if (n < MIN_LEN) return ERROR;\n  target(n);\n}",
            },
        })
        verifier, _ = _make_verifier(index)
        result = verifier._find_sibling_call_sites("some/file.c:target")
        assert "AUTOMATED CALLER CHECK" in result
        assert "1 other call site" in result
        assert "some/other.c:caller" in result
        assert "if (n < MIN_LEN) return ERROR;" in result

    def test_excludes_the_flagged_function_itself(self):
        # Recursive call: target calls itself. Must not be listed as its
        # own "sibling" — that's not evidence of an independent guard.
        index = _index_with({
            "some/file.c:target": {
                "name": "target",
                "code": "void target(int n) { if (n > 0) target(n - 1); }",
            },
        })
        verifier, _ = _make_verifier(index)
        assert verifier._find_sibling_call_sites("some/file.c:target") == ""

    def test_falls_back_to_name_split_when_function_id_not_in_index(self):
        # function_id passed in doesn't exactly match an index key (e.g.
        # route_key uses a different convention than RepositoryIndex) —
        # falls back to the segment after the last ":" as the bare name.
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) {}"},
            "some/other.c:caller": {"name": "caller", "code": "void caller() { target(5); }"},
        })
        verifier, _ = _make_verifier(index)
        result = verifier._find_sibling_call_sites("mismatched/key/format:target")
        assert "some/other.c:caller" in result

    def test_bounded_to_five_callers_with_remainder_note(self):
        functions = {"some/file.c:target": {"name": "target", "code": "void target() {}"}}
        for i in range(7):
            functions[f"some/caller{i}.c:caller{i}"] = {
                "name": f"caller{i}", "code": f"void caller{i}() {{ target(); }}",
            }
        index = _index_with(functions)
        verifier, _ = _make_verifier(index)
        result = verifier._find_sibling_call_sites("some/file.c:target")
        shown = result.count("--- Caller:")
        assert shown == 5
        assert "and 2 more call site(s)" in result

    def test_search_usages_exception_returns_empty_not_raises(self, monkeypatch):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target() {}"},
        })
        verifier, _ = _make_verifier(index)
        monkeypatch.setattr(index, "search_usages", lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
        assert verifier._find_sibling_call_sites("some/file.c:target") == ""


class TestVerifyResultInjectsSiblingContext:
    def test_prompt_includes_caller_context_when_function_id_given(self):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) { do_thing(n); }"},
            "some/other.c:caller": {
                "name": "caller",
                "code": "void caller(int n) {\n  if (n < MIN_LEN) return ERROR;\n  target(n);\n}",
            },
        })
        verifier, adapter = _make_verifier(index)
        verifier.verify_result(
            code="void target(int n) { do_thing(n); }",
            finding="vulnerable", attack_vector="oob read", reasoning="no length check",
            function_id="some/file.c:target",
        )
        first_call = adapter.calls[0]
        prompt_text = first_call["messages"][0].content[0].text
        assert "AUTOMATED CALLER CHECK" in prompt_text
        assert "if (n < MIN_LEN) return ERROR;" in prompt_text

    def test_prompt_unchanged_when_no_function_id_given(self):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) { do_thing(n); }"},
            "some/other.c:caller": {"name": "caller", "code": "void caller(int n) { target(n); }"},
        })
        verifier, adapter = _make_verifier(index)
        verifier.verify_result(
            code="void target(int n) { do_thing(n); }",
            finding="vulnerable", attack_vector="oob read", reasoning="no length check",
        )
        prompt_text = adapter.calls[0]["messages"][0].content[0].text
        assert "AUTOMATED CALLER CHECK" not in prompt_text

    def test_prompt_unchanged_when_no_siblings_exist(self):
        index = _index_with({
            "some/file.c:target": {"name": "target", "code": "void target(int n) { do_thing(n); }"},
        })
        verifier, adapter = _make_verifier(index)
        verifier.verify_result(
            code="void target(int n) { do_thing(n); }",
            finding="vulnerable", attack_vector="oob read", reasoning="no length check",
            function_id="some/file.c:target",
        )
        prompt_text = adapter.calls[0]["messages"][0].content[0].text
        assert "AUTOMATED CALLER CHECK" not in prompt_text
