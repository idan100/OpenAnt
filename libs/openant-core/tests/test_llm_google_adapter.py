"""Google-adapter-specific tests (PR #69 fixes C1 + H1).

* C1 — Gemini matches a ``function_response`` to its ``function_call``
  by NAME, not id. The pipeline now carries the originating tool's name
  on ``ToolResultBlock.name``; the adapter must send THAT as the
  function_response name, not the synthesised ``gemini_<name>_<idx>`` id.
* H1 — a 429 reports to the process-global rate limiter so sibling
  workers back off.
"""

from __future__ import annotations

import pytest

from utilities.llm import LLMRateLimitError, LLMResponseError, Message, TextBlock, ToolDef, ToolResultBlock, ToolUseBlock
from utilities.llm.providers.google import (
    _message_to_gemini,
    _name_for_tool_result,
    _response_to_unified,
    _sanitize_schema_for_gemini,
    _tool_to_gemini,
)
from utilities.llm_client import reset_warning_state
from utilities.rate_limiter import get_rate_limiter, is_retryable_error, reset_rate_limiter


@pytest.fixture(autouse=True)
def _reset_state():
    reset_rate_limiter()
    reset_warning_state()
    yield
    reset_rate_limiter()
    reset_warning_state()


# ---------------------------------------------------------------------------
# C1 — function name survives the round trip
# ---------------------------------------------------------------------------


def test_name_for_tool_result_prefers_name():
    # When the pipeline supplies the originating tool name, use it.
    block = ToolResultBlock(tool_use_id="gemini_search_code_0", name="search_code", content="x")
    assert _name_for_tool_result(block) == "search_code"


def test_name_for_tool_result_falls_back_to_id_when_no_name():
    block = ToolResultBlock(tool_use_id="legacy_id", content="x")
    assert _name_for_tool_result(block) == "legacy_id"


def test_function_response_carries_function_name():
    """The whole point of C1: the function_response Part Gemini receives
    must be named after the original function (``search_code``), not the
    synthesised id (``gemini_search_code_0``) — otherwise Gemini can't
    match the result to its call."""
    msg = Message(
        role="user",
        content=[ToolResultBlock(
            tool_use_id="gemini_search_code_0",
            name="search_code",
            content='{"hits": 1}',
        )],
    )
    content = _message_to_gemini(msg)
    part = content.parts[0]
    assert part.function_response is not None
    assert part.function_response.name == "search_code", (
        "C1: Gemini matches function_response to function_call by NAME; "
        "sending the synthesised id would never match the original call"
    )


# ---------------------------------------------------------------------------
# thought_signature — required on replay by Gemini's thinking models
# (2.5+/3.x), or the second turn 400s with "Function call is missing a
# thought_signature". Missed originally because the SDK attaches this
# to the PART carrying the function_call, not to the FunctionCall
# object nested inside it.
# ---------------------------------------------------------------------------


def test_response_to_unified_captures_thought_signature():
    from tests._llm_factories.google import _candidate, _function_call_part, _response

    response = _response(
        candidates=[_candidate(
            parts=[_function_call_part(
                name="echo", args={"text": "hi"}, id="gemini_echo_0",
                thought_signature=b"sig-bytes-123",
            )],
        )],
        prompt_tokens=10, candidate_tokens=8,
    )
    result = _response_to_unified(response)
    tool_use = result.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.thought_signature == b"sig-bytes-123"


def test_response_to_unified_tolerates_missing_thought_signature():
    """A function_call part with no thought_signature attribute at all
    (older SDK behavior, or a non-thinking model) must not crash."""
    from types import SimpleNamespace
    from tests._llm_factories.google import _candidate, _response

    fc = SimpleNamespace(name="echo", args={"text": "hi"}, id="gemini_echo_0")
    part = SimpleNamespace(text=None, function_call=fc)  # no .thought_signature at all
    response = _response(candidates=[_candidate(parts=[part])], prompt_tokens=10, candidate_tokens=8)
    result = _response_to_unified(response)
    assert result.content[0].thought_signature is None


def test_message_to_gemini_replays_thought_signature_on_the_part():
    """The actual bug: the signature must land on the outgoing Part
    (Part.from_function_call doesn't accept it as a kwarg), not get
    silently dropped on replay."""
    msg = Message(
        role="assistant",
        content=[ToolUseBlock(
            id="gemini_echo_0", name="echo", input={"text": "hi"},
            thought_signature=b"sig-bytes-123",
        )],
    )
    content = _message_to_gemini(msg)
    assert content.parts[0].thought_signature == b"sig-bytes-123"


def test_message_to_gemini_omits_thought_signature_when_absent():
    """A ToolUseBlock from a non-Gemini origin (or built by hand in a
    test) has no signature — must not send a bogus one."""
    msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="t_1", name="echo", input={"text": "hi"})],
    )
    content = _message_to_gemini(msg)
    assert content.parts[0].thought_signature is None


# ---------------------------------------------------------------------------
# Nullable-union tool schema fields ("type": ["string", "null"], valid
# JSON Schema 2020-12) previously crashed FunctionDeclaration construction
# outright for ANY tool using this pattern — e.g. FindingVerifier's
# ``finish`` tool (exploit_path.entry_point, .path_broken_at,
# security_weakness). Gemini's Schema.type only accepts a single scalar.
# ---------------------------------------------------------------------------


def test_sanitize_schema_converts_nullable_union_to_nullable_flag():
    schema = {"type": ["string", "null"], "description": "x"}
    result = _sanitize_schema_for_gemini(schema)
    assert result["type"] == "string"
    assert result["nullable"] is True


def test_sanitize_schema_recurses_into_properties_and_items():
    schema = {
        "type": "object",
        "properties": {
            "entry_point": {"type": ["string", "null"]},
            "data_flow": {"type": "array", "items": {"type": ["string", "null"]}},
            "sink_reached": {"type": "boolean"},
        },
    }
    result = _sanitize_schema_for_gemini(schema)
    assert result["properties"]["entry_point"]["type"] == "string"
    assert result["properties"]["entry_point"]["nullable"] is True
    assert result["properties"]["data_flow"]["items"]["type"] == "string"
    assert result["properties"]["data_flow"]["items"]["nullable"] is True
    assert result["properties"]["sink_reached"]["type"] == "boolean"
    assert "nullable" not in result["properties"]["sink_reached"]


def test_sanitize_schema_leaves_ordinary_schemas_untouched():
    schema = {"type": "string", "enum": ["a", "b"]}
    assert _sanitize_schema_for_gemini(schema) == schema


def test_sanitize_schema_does_not_mutate_input():
    schema = {"type": ["string", "null"]}
    _sanitize_schema_for_gemini(schema)
    assert schema == {"type": ["string", "null"]}, "must not mutate the caller's ToolDef schema"


def test_tool_to_gemini_builds_finding_verifier_finish_tool_without_crashing():
    """Reproduces the exact real-world crash: FindingVerifier's ``finish``
    tool schema, which is what actually broke every Stage 2 Gemini call."""
    from utilities.finding_verifier import VERIFICATION_TOOLS

    finish = next(t for t in VERIFICATION_TOOLS if t["name"] == "finish")
    tool = ToolDef(name=finish["name"], description=finish.get("description", ""), input_schema=finish["input_schema"])

    gemini_tool = _tool_to_gemini(tool)  # must not raise

    params = gemini_tool.function_declarations[0].parameters
    entry_point = params.properties["exploit_path"].properties["entry_point"]
    assert str(entry_point.type) in ("Type.STRING", "STRING")
    assert entry_point.nullable is True


# ---------------------------------------------------------------------------
# Gemini tool-calling failure states (MALFORMED_FUNCTION_CALL /
# UNEXPECTED_TOOL_CALL) — provider-reported failures, must not silently
# normalise to a clean end_turn like an ordinary unknown finish_reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
def test_tool_error_finish_reasons_raise_instead_of_normalising(reason):
    from tests._llm_factories.google import _candidate, _response

    response = _response(
        candidates=[_candidate(parts=[], finish_reason=reason)],
        prompt_tokens=1, candidate_tokens=0,
    )
    with pytest.raises(LLMResponseError) as excinfo:
        _response_to_unified(response)
    # Must be retryable — a fresh retry gets a new conversation, which can
    # land on a different pool candidate entirely.
    assert is_retryable_error(str(excinfo.value))


def test_normal_finish_reason_still_normalises():
    from tests._llm_factories.google import _candidate, _response, _text_part

    response = _response(
        candidates=[_candidate(parts=[_text_part("hi")], finish_reason="STOP")],
        prompt_tokens=1, candidate_tokens=1,
    )
    result = _response_to_unified(response)
    assert result.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Explicit HTTP timeout — the SDK's own default is unset, which httpx
# treats as "disable all timeouts" rather than falling back to its own
# 5s default. Without this, a stalled connection or a very slow
# generation hangs the adapter forever (observed: 220+ seconds with
# zero progress on an app_context call, until manually interrupted).
# ---------------------------------------------------------------------------


def test_client_constructed_with_explicit_timeout(monkeypatch):
    from utilities.llm.providers import google as google_module

    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(google_module.genai, "Client", _FakeClient)
    google_module.GoogleAdapter(api_key="x")

    http_options = captured.get("http_options")
    assert http_options is not None, "must always construct HttpOptions to carry the timeout"
    assert http_options.timeout == 600_000, "600s in milliseconds, matching Anthropic/OpenAI's read timeout default"


def test_timeout_is_set_even_with_no_other_http_options_needed(monkeypatch):
    """Previously HttpOptions was only constructed when base_url or a
    non-default max_retries needed it -- confirm the timeout survives
    even on the plain-default construction path."""
    from utilities.llm.providers import google as google_module

    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(google_module.genai, "Client", _FakeClient)
    google_module.GoogleAdapter(api_key="x", base_url=None, max_retries=None)

    assert captured.get("http_options") is not None
    assert captured["http_options"].timeout == 600_000


# ---------------------------------------------------------------------------
# H1 — rate-limiter coordination
# ---------------------------------------------------------------------------


def test_rate_limit_reports_to_global_limiter():
    from tests._llm_factories.google import make_adapter

    adapter = make_adapter("rate_limit")  # scripted to raise a 429 (retry_after=7)
    limiter = get_rate_limiter("google")
    assert not limiter.is_in_backoff()
    with pytest.raises(LLMRateLimitError):
        adapter.complete(
            model="gemini-2.5-pro",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )
    assert limiter.is_in_backoff(), "Google 429 must trigger global backoff (H1)"
