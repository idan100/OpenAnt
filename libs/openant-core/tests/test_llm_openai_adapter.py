"""OpenAI-adapter-specific tests (PR #69 fixes H1 + H3 + L2 + L3).

The shared contract harness (``test_llm_adapter_contract.py``) covers
behaviors every adapter must satisfy. This file covers OpenAI specifics:

* H3 — reasoning models (o1/o3/o4) send ``max_completion_tokens``, not
  ``max_tokens``; regular chat models (gpt-4o) keep ``max_tokens``. Also,
  reasoning models reject the ``system`` role, so a system prompt is
  routed to a ``developer``-role message; non-reasoning models keep
  ``system``. o1-mini/o1-preview are dropped entirely (no tool support).
* H1 — a 429 reports to the process-global rate limiter so sibling
  workers back off, and ``complete()`` consults the limiter first.
* L2 — an empty ``choices`` array surfaces ``LLMResponseError`` instead
  of letting an ``IndexError`` escape the taxonomy.
* L3 — the pricing table carries current models so calls don't silently
  report $0.

These stub the SDK boundary so nothing hits the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from utilities.llm import (
    LLMRateLimitError,
    LLMResponseError,
    Message,
    TextBlock,
    ToolDef,
    ToolUseBlock,
)
from utilities.llm.providers.openai import OpenAIAdapter, _messages_to_openai, reset_warnings
from utilities.llm_client import reset_warning_state
from utilities.rate_limiter import get_rate_limiter, is_retryable_error, reset_rate_limiter


@pytest.fixture(autouse=True)
def _reset_state():
    # Once OpenAI wires into the global limiter, a leaked backoff would
    # make later tests sleep ~30s. Reset before and after every test.
    reset_rate_limiter()
    reset_warning_state()
    reset_warnings()
    yield
    reset_rate_limiter()
    reset_warning_state()
    reset_warnings()


def _text_response(*, prompt_tokens=1, completion_tokens=1):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _stub(side_effect):
    client = MagicMock(spec=openai.OpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=side_effect)
    return OpenAIAdapter(_client=client), client


def _fake_http(status, *, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    return httpx.Response(
        status_code=status,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


def _hi():
    return [Message(role="user", content=[TextBlock("hi")])]


# ---------------------------------------------------------------------------
# H3 — reasoning models need max_completion_tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "o3", "openai/o1"])
def test_reasoning_model_uses_max_completion_tokens(model):
    adapter, client = _stub(lambda **kw: _text_response())
    adapter.complete(model=model, system=None, messages=_hi(), max_tokens=64)
    kw = client.chat.completions.create.call_args.kwargs
    assert kw.get("max_completion_tokens") == 64
    assert "max_tokens" not in kw, f"{model}: reasoning models reject max_tokens"


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
def test_chat_model_uses_max_tokens(model):
    adapter, client = _stub(lambda **kw: _text_response())
    adapter.complete(model=model, system=None, messages=_hi(), max_tokens=64)
    kw = client.chat.completions.create.call_args.kwargs
    assert kw.get("max_tokens") == 64
    assert "max_completion_tokens" not in kw


def test_validate_reasoning_model_uses_max_completion_tokens():
    adapter, client = _stub(lambda **kw: _text_response())
    adapter.validate("o3-mini")
    kw = client.chat.completions.create.call_args.kwargs
    assert kw.get("max_completion_tokens") == 1
    assert "max_tokens" not in kw


# ---------------------------------------------------------------------------
# H3 — reasoning models reject the ``system`` role → route to ``developer``
# ---------------------------------------------------------------------------


def _roles(client) -> list[str]:
    """Roles, in order, of the messages sent on the last create() call."""
    kw = client.chat.completions.create.call_args.kwargs
    return [m["role"] for m in kw["messages"]]


@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "openai/o1"])
def test_reasoning_model_routes_system_to_developer(model):
    adapter, client = _stub(lambda **kw: _text_response())
    adapter.complete(
        model=model, system="be careful", messages=_hi(), max_tokens=8
    )
    kw = client.chat.completions.create.call_args.kwargs
    roles = [m["role"] for m in kw["messages"]]
    assert "developer" in roles, f"{model}: reasoning models need a developer role"
    assert "system" not in roles, f"{model}: reasoning models reject the system role"
    dev = next(m for m in kw["messages"] if m["role"] == "developer")
    assert dev["content"] == "be careful"


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
def test_chat_model_keeps_system_role(model):
    adapter, client = _stub(lambda **kw: _text_response())
    adapter.complete(
        model=model, system="be careful", messages=_hi(), max_tokens=8
    )
    roles = _roles(client)
    assert "system" in roles, f"{model}: non-reasoning models keep the system role"
    assert "developer" not in roles
    kw = client.chat.completions.create.call_args.kwargs
    sysmsg = next(m for m in kw["messages"] if m["role"] == "system")
    assert sysmsg["content"] == "be careful"


def test_dropped_reasoning_models_absent_from_pricing():
    # o1-mini / o1-preview reject the developer role AND lack tool support,
    # so the adapter no longer advertises them (H3).
    assert "o1-mini" not in OpenAIAdapter.pricing
    assert "o1-preview" not in OpenAIAdapter.pricing
    # The reasoning models we DO keep stay priced.
    assert "o1" in OpenAIAdapter.pricing
    assert "o3-mini" in OpenAIAdapter.pricing


# ---------------------------------------------------------------------------
# L2 — empty ``choices`` surfaces LLMResponseError (not a bare IndexError)
# ---------------------------------------------------------------------------


def test_empty_choices_raises_llm_response_error():
    empty = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
    )
    adapter, _ = _stub(lambda **kw: empty)
    with pytest.raises(LLMResponseError):
        adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)


# ---------------------------------------------------------------------------
# L3 — pricing table carries current models so they don't report $0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"]
)
def test_current_models_present_in_pricing(model):
    rates = OpenAIAdapter.pricing.get(model)
    assert rates is not None, f"{model}: must be priced so it doesn't report $0"
    assert rates["input"] > 0 and rates["output"] > 0


# ---------------------------------------------------------------------------
# H1 — rate-limiter coordination
# ---------------------------------------------------------------------------


def test_rate_limit_reports_to_global_limiter():
    def boom(**kw):
        raise openai.RateLimitError(
            message="slow down", response=_fake_http(429, retry_after="7"), body=None
        )

    adapter, _ = _stub(boom)
    limiter = get_rate_limiter("openai")
    assert not limiter.is_in_backoff()
    with pytest.raises(LLMRateLimitError):
        adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)
    assert limiter.is_in_backoff(), "OpenAI 429 must trigger global backoff (H1)"


def test_complete_consults_limiter_before_request(monkeypatch):
    adapter, _ = _stub(lambda **kw: _text_response())
    seen = {"waited": False}
    limiter = get_rate_limiter("openai")
    monkeypatch.setattr(
        limiter, "wait_if_needed", lambda: (seen.__setitem__("waited", True), 0.0)[1]
    )
    adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)
    assert seen["waited"], "complete() must call wait_if_needed before the request (H1)"


# ---------------------------------------------------------------------------
# Cloudflare "oneOf not met ... 'string' not in 'null'" — an assistant
# tool-call-only message must send content="" not content=None. Confirmed
# empirically against the real Cloudflare Workers AI API.
# ---------------------------------------------------------------------------


def test_assistant_tool_call_only_message_sends_empty_string_content():
    messages = [
        Message(role="user", content=[TextBlock("call it")]),
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="echo", input={})]),
    ]
    out = _messages_to_openai(messages, system=None, model="gpt-4o")
    assistant_msg = next(m for m in out if m["role"] == "assistant")
    assert assistant_msg["content"] == "", "must be empty string, not None (Cloudflare 400s on null)"


def test_assistant_message_with_text_and_tool_calls_keeps_the_text():
    messages = [
        Message(role="assistant", content=[
            TextBlock("let me check"),
            ToolUseBlock(id="t1", name="echo", input={}),
        ]),
    ]
    out = _messages_to_openai(messages, system=None, model="gpt-4o")
    assert out[0]["content"] == "let me check"


# ---------------------------------------------------------------------------
# finish_reason='error' / 'tool_use_failed' — provider-reported failures
# must not silently normalise to a clean end_turn.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["error", "tool_use_failed"])
def test_error_finish_reasons_raise_instead_of_normalising(reason):
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=None),
            finish_reason=reason,
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
    )
    adapter, _ = _stub(lambda **kw: response)
    with pytest.raises(LLMResponseError) as excinfo:
        adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)
    # Must be retryable — a fresh retry gets a new conversation, which can
    # land on a different pool candidate entirely.
    assert is_retryable_error(str(excinfo.value))


def test_normal_finish_reasons_still_pass_through():
    adapter, _ = _stub(lambda **kw: _text_response())
    result = adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)
    assert result.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Hallucinated tool names — visibility (one-time warning), not a behavior
# change: ToolExecutor already handles an unknown tool name gracefully.
# ---------------------------------------------------------------------------


def _tool_call_response(name, arguments="{}"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="t1", function=SimpleNamespace(name=name, arguments=arguments),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def test_hallucinated_tool_call_warns_once(capsys):
    adapter, _ = _stub(lambda **kw: _tool_call_response("not_offered"))
    tools = [ToolDef(name="search_usages", description="x", input_schema={})]
    adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8, tools=tools)
    err = capsys.readouterr().err
    assert "not_offered" in err
    assert "not in the offered tools list" in err


def test_offered_tool_call_does_not_warn(capsys):
    adapter, _ = _stub(lambda **kw: _tool_call_response("search_usages"))
    tools = [ToolDef(name="search_usages", description="x", input_schema={})]
    adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8, tools=tools)
    err = capsys.readouterr().err
    assert "hallucinated" not in err.lower()


def test_hallucinated_tool_still_produces_a_usable_tool_use_block():
    """Visibility only — ToolExecutor downstream still gets a normal
    ToolUseBlock so its existing 'unknown tool' self-correction flow
    keeps working unchanged."""
    adapter, _ = _stub(lambda **kw: _tool_call_response("not_offered"))
    tools = [ToolDef(name="search_usages", description="x", input_schema={})]
    result = adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8, tools=tools)
    tool_use = next(b for b in result.content if isinstance(b, ToolUseBlock))
    assert tool_use.name == "not_offered"


# ---------------------------------------------------------------------------
# Pseudo-tool-call syntax leakage (<function=...></function> in plain
# text instead of a native tool_calls entry) — observed from weaker
# open-source models on some free-tier pools.
# ---------------------------------------------------------------------------


def test_pseudo_tool_call_syntax_raises_retryable_error():
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content='<function=search_usages{"function_name": "x"}</function>',
                tool_calls=None,
            ),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    adapter, _ = _stub(lambda **kw: response)
    tools = [ToolDef(name="search_usages", description="x", input_schema={})]
    with pytest.raises(LLMResponseError) as excinfo:
        adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8, tools=tools)
    assert is_retryable_error(str(excinfo.value))


def test_pseudo_tool_call_syntax_ignored_when_no_tools_were_offered():
    """The same text is fine as plain output when tools weren't even
    on the table — only flagged when it looks like a failed attempt to
    use tools that WERE offered."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content='the docs show <function=foo></function> as an example',
                tool_calls=None,
            ),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    adapter, _ = _stub(lambda **kw: response)
    result = adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8)  # no tools=
    assert result.stop_reason == "end_turn"


def test_normal_text_response_with_tools_offered_is_unaffected():
    adapter, _ = _stub(lambda **kw: _text_response())
    tools = [ToolDef(name="search_usages", description="x", input_schema={})]
    result = adapter.complete(model="gpt-4o", system=None, messages=_hi(), max_tokens=8, tools=tools)
    assert result.stop_reason == "end_turn"
