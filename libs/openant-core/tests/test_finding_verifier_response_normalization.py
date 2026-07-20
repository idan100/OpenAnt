"""Tests for FindingVerifier's defense against malformed LLM-response
shapes: ``result.get("verification", {}).get(...)`` chains crash with
``'str' object has no attribute 'get'`` when a weaker model writes e.g.
``"verification": "N/A"`` or ``"exploit_path": "not applicable"``
instead of an object — the key IS present, so ``.get(key, {})``'s
default never kicks in. ``_safe_dict`` (and its use at every such
chain) fixes this by treating "present but not a dict" the same as
"absent" rather than crashing.
"""

from __future__ import annotations

from utilities.agentic_enhancer.repository_index import RepositoryIndex
from utilities.finding_verifier import FindingVerifier, _safe_dict
from utilities.llm import PhaseBinding


class _StubAdapter:
    name = "stub"
    supports_tools = True

    def complete(self, **kwargs):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def validate(self, model):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _verifier() -> FindingVerifier:
    index = RepositoryIndex({"functions": {}}, repo_path=None)
    binding = PhaseBinding(phase="verify", adapter=_StubAdapter(), model="m", provider_name="stub")
    return FindingVerifier(index=index, binding=binding)


# ---------------------------------------------------------------------------
# _safe_dict itself
# ---------------------------------------------------------------------------


def test_safe_dict_passes_through_real_dicts():
    d = {"a": 1}
    assert _safe_dict(d) is d


def test_safe_dict_treats_non_dict_as_absent():
    assert _safe_dict("N/A") == {}
    assert _safe_dict(None) == {}
    assert _safe_dict(["not", "a", "dict"]) == {}
    assert _safe_dict(42) == {}


# ---------------------------------------------------------------------------
# _has_conclusive_exploit_path — the exact reported crash shape
# ---------------------------------------------------------------------------


def test_has_conclusive_exploit_path_survives_string_verification():
    verifier = _verifier()
    result = {"verification": "N/A"}  # observed shape from a weaker model
    assert verifier._has_conclusive_exploit_path(result) is False


def test_has_conclusive_exploit_path_survives_string_exploit_path():
    verifier = _verifier()
    result = {"verification": {"exploit_path": "not applicable"}}
    assert verifier._has_conclusive_exploit_path(result) is False


def test_has_conclusive_exploit_path_still_works_for_real_dicts():
    verifier = _verifier()
    result = {"verification": {"exploit_path": {"sink_reached": False}}}
    assert verifier._has_conclusive_exploit_path(result) is True


def test_has_conclusive_exploit_path_handles_missing_verification_entirely():
    verifier = _verifier()
    assert verifier._has_conclusive_exploit_path({}) is False


# ---------------------------------------------------------------------------
# _check_consistency's verdict-extraction chains
# ---------------------------------------------------------------------------


def test_check_consistency_survives_string_verification_field():
    verifier = _verifier()
    # "errorMsg"/"infoMsg" both normalize to the same "*Msg" pattern key
    # in _group_by_pattern, landing in one group of 2 — the shape needed
    # to actually reach the .get("verification", {}).get(...) chain at
    # all (a group of 1 short-circuits before it). SAME finding on both
    # keeps verdicts consistent so _resolve_inconsistency (a real LLM
    # call) never fires — this test is purely about surviving the
    # extraction chain, not the resolution flow.
    results = [
        {"route_key": "a.js:errorMsg", "finding": "safe", "verification": "N/A"},
        {"route_key": "a.js:infoMsg", "finding": "safe", "verification": "N/A"},
    ]
    # Must not raise — this previously crashed with
    # 'str' object has no attribute 'get' on the .get("verification", {}).get(...) chain.
    out = verifier._check_consistency(results, code_by_route={})
    assert out is results
