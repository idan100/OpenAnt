"""Tests for the generic Stage 1 "inconclusive" resolution pass
(utilities/stage1_inconclusive_resolver.py).

Two outcomes it should produce deterministically/correctly:
- A function with no callers anywhere in the analyzed codebase is dead
  code -> demoted straight to "safe", no LLM call.
- A function WITH callers gets one re-analysis attempt with caller code
  appended; if that resolves the ambiguity, the verdict is replaced.

The dead-code check itself has two layers: RepositoryIndex.search_usages
only sees callers inside OTHER DECLARED FUNCTIONS, so a set of tests
below cover the raw-source fallback that catches callers sitting in
top-level/module-scope script code instead.
"""

from __future__ import annotations

import pytest

from utilities.agentic_enhancer.repository_index import RepositoryIndex
from utilities.stage1_inconclusive_resolver import resolve_inconclusive_findings


def _index_with(functions: dict) -> RepositoryIndex:
    return RepositoryIndex({"functions": functions}, repo_path=None)


def test_noop_when_index_is_none():
    results = [{"unit_id": "a.js:f", "finding": "inconclusive"}]
    out = resolve_inconclusive_findings(results, {}, binding=None, index=None)
    assert out[0]["finding"] == "inconclusive"
    assert "stage1_inconclusive_resolution" not in out[0]


def test_demotes_dead_code_with_no_callers():
    index = _index_with({
        "a.js:loadConfig": {"name": "loadConfig", "code": "function loadConfig(x) { return x; }"},
    })
    results = [{"unit_id": "a.js:loadConfig", "finding": "inconclusive", "route_key": "a.js:loadConfig"}]
    code_by_route = {"a.js:loadConfig": "function loadConfig(x) { return x; }"}

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "safe"
    assert out[0]["verdict"] == "safe"
    assert out[0]["stage1_inconclusive_resolution"] == "demoted_dead_code"


def test_resolves_via_caller_context(monkeypatch):
    index = _index_with({
        "a.js:loadConfig": {"name": "loadConfig", "code": "function loadConfig(x) { return x; }"},
        "a.js:main": {"name": "main", "code": 'loadConfig("hardcoded");'},
    })
    results = [{"unit_id": "a.js:loadConfig", "finding": "inconclusive", "route_key": "a.js:loadConfig"}]
    code_by_route = {"a.js:loadConfig": "function loadConfig(x) { return x; }"}

    def fake_simple_text(binding, prompt, system=None):
        assert "Caller: a.js:main" in prompt
        return '{"finding": "safe", "reasoning": "only called with a hardcoded value"}'

    monkeypatch.setattr("utilities.llm.simple_text", fake_simple_text)

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "safe"
    assert out[0]["unit_id"] == "a.js:loadConfig"  # preserved, not overwritten
    assert out[0]["stage1_inconclusive_resolution"] == "resolved_via_caller_context"


def test_leaves_still_inconclusive_alone(monkeypatch):
    index = _index_with({
        "a.js:loadConfig": {"name": "loadConfig", "code": "function loadConfig(x) { return x; }"},
        "a.js:main": {"name": "main", "code": "loadConfig(userInput);"},
    })
    results = [{"unit_id": "a.js:loadConfig", "finding": "inconclusive", "route_key": "a.js:loadConfig"}]
    code_by_route = {"a.js:loadConfig": "function loadConfig(x) { return x; }"}

    def fake_simple_text(binding, prompt, system=None):
        return '{"finding": "inconclusive", "reasoning": "still can\'t trace userInput\'s origin"}'

    monkeypatch.setattr("utilities.llm.simple_text", fake_simple_text)

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "inconclusive"
    assert "stage1_inconclusive_resolution" not in out[0]


# ---------------------------------------------------------------------------
# Raw-source fallback: catches callers RepositoryIndex.search_usages can't
# see because they live in top-level/module-scope code, not another
# declared function.
# ---------------------------------------------------------------------------


def test_does_not_demote_when_only_caller_is_top_level_code(tmp_path, monkeypatch):
    (tmp_path / "a.js").write_text(
        "function main(product) {\n"
        "  return loadFile(product);\n"
        "}\n"
        "\n"
        "const items = ['a', 'b'];\n"
        "items.forEach((item) => {\n"
        "  main(item);\n"
        "});\n",
        encoding="utf-8",
    )
    index = RepositoryIndex(
        {"functions": {"a.js:main": {"name": "main", "code": "function main(product) {\n  return loadFile(product);\n}", "startLine": 1, "endLine": 3}}},
        repo_path=str(tmp_path),
    )
    results = [{"unit_id": "a.js:main", "finding": "inconclusive", "route_key": "a.js:main"}]
    code_by_route = {"a.js:main": "function main(product) {\n  return loadFile(product);\n}"}

    def fake_simple_text(binding, prompt, system=None):
        assert "top-level/module-scope code" in prompt
        assert "items.forEach" in prompt
        return '{"finding": "safe", "reasoning": "only called with hardcoded literals from the forEach loop"}'

    monkeypatch.setattr("utilities.llm.simple_text", fake_simple_text)

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "safe"
    assert out[0]["stage1_inconclusive_resolution"] == "resolved_via_caller_context"


def test_still_demotes_when_no_caller_anywhere_including_top_level(tmp_path):
    (tmp_path / "a.js").write_text(
        "function main(product) {\n  return loadFile(product);\n}\n",
        encoding="utf-8",
    )
    index = RepositoryIndex(
        {"functions": {"a.js:main": {"name": "main", "code": "function main(product) {\n  return loadFile(product);\n}", "startLine": 1, "endLine": 3}}},
        repo_path=str(tmp_path),
    )
    results = [{"unit_id": "a.js:main", "finding": "inconclusive", "route_key": "a.js:main"}]
    code_by_route = {"a.js:main": "function main(product) {\n  return loadFile(product);\n}"}

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "safe"
    assert out[0]["stage1_inconclusive_resolution"] == "demoted_dead_code"


def test_raw_source_check_skipped_gracefully_when_repo_path_missing():
    """No repo_path -> can't grep the file -> falls back to today's
    behavior (demote on empty search_usages) rather than erroring."""
    index = _index_with({
        "a.js:main": {"name": "main", "code": "function main(product) {}", "startLine": 1, "endLine": 1},
    })
    results = [{"unit_id": "a.js:main", "finding": "inconclusive", "route_key": "a.js:main"}]
    code_by_route = {"a.js:main": "function main(product) {}"}

    out = resolve_inconclusive_findings(results, code_by_route, binding=object(), index=index)

    assert out[0]["finding"] == "safe"
    assert out[0]["stage1_inconclusive_resolution"] == "demoted_dead_code"
