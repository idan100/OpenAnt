"""ClaudeSubscriptionAdapter-specific tests.

This adapter is NOT registered in the shared contract harness
(``test_llm_adapter_contract.py``) — its capability profile genuinely
differs from the metered adapters in two ways the shared contract
hard-asserts on:

* ``retry_after`` on a rate-limit error: the raw provider adapters get
  this from an HTTP ``retry-after`` header; the Agent SDK's
  ``AssistantMessage.error`` is a bare string with no such field.
  This adapter reads one opportunistically (see
  ``providers/claude_subscription.py::_translate``) but has nothing to
  supply for real traffic today.
* ``LLMNotFoundError`` on a bad model: the SDK's error literal has no
  ``model_not_found``-shaped value distinct from generic
  ``invalid_request`` — there's no honest way to tell them apart.

Forcing this adapter into the shared parametrize would mean scripting
its fake to claim capabilities the real SDK doesn't have. This file
covers the same spirit (auth/rate-limit/connection/empty-response
mapping, tool-use round trip, validate()) against what the adapter
ACTUALLY does, plus the two behaviors unique to it: transcript
flattening and ``max_output_tokens`` → ``stop_reason="max_tokens"``.

Nothing here imports the real ``claude_agent_sdk`` package (it's an
optional dependency) — the constructor's import check is satisfied by
inserting a bare stub module into ``sys.modules``, and the actual
SDK interaction is stubbed by monkeypatching the module-level
``_run_query`` / ``_build_tool_bridge`` functions.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from utilities.llm import (
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)
from utilities.llm.providers import claude_subscription as mod
from utilities.rate_limiter import get_rate_limiter, reset_rate_limiter


# Stand-ins for the real claude_agent_sdk content-block dataclasses.
# IMPORTANT: the real SDK's TextBlock/ToolUseBlock/ThinkingBlock carry NO
# ``.type`` discriminator — utilities/llm/providers/claude_subscription.py
# dispatches on them via isinstance(), confirmed against the installed
# package (dataclasses.fields(ThinkingBlock) == ('thinking', 'signature')).
# These fakes exist so tests don't need the real (optional) package
# installed, while still exercising the real isinstance-based dispatch —
# `_translate`/`_has_substantive_content` import these exact classes off
# the stubbed ``claude_agent_sdk`` module (see the autouse fixture below).
class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input


class _FakeThinkingBlock:
    def __init__(self, thinking="...", signature=""):
        self.thinking = thinking
        self.signature = signature


class _FakeAssistantMessage:
    def __init__(self, content):
        self.content = content


class _FakeResultMessage:
    def __init__(self, subtype="success"):
        self.subtype = subtype


class _FakeRateLimitEvent:
    def __init__(self, rate_limit_info):
        self.rate_limit_info = rate_limit_info


@pytest.fixture(autouse=True)
def _stub_claude_agent_sdk_module(monkeypatch):
    """Make ``import claude_agent_sdk`` succeed inside the constructor
    without the real (optional) package installed, and give ``_translate``
    / ``_has_substantive_content`` / ``_run_query`` the block/message
    classes they isinstance()-check against. ``query`` itself is left
    unset here — only the ``TestRunQueryExceptionSalvage`` class (which
    exercises the real ``_run_query``, not the ``_stub_run_query``
    replacement every other test uses) monkeypatches it in.
    """
    fake_mod = types.ModuleType("claude_agent_sdk")
    fake_mod.TextBlock = _FakeTextBlock
    fake_mod.ToolUseBlock = _FakeToolUseBlock
    fake_mod.ThinkingBlock = _FakeThinkingBlock
    fake_mod.AssistantMessage = _FakeAssistantMessage
    fake_mod.ResultMessage = _FakeResultMessage
    fake_mod.RateLimitEvent = _FakeRateLimitEvent
    fake_mod.ClaudeAgentOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)
    return fake_mod


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture(autouse=True)
def _reset_utilization_tracker():
    mod._UtilizationTracker.reset()
    yield
    mod._UtilizationTracker.reset()


def _assistant(content, *, usage=None, error=None, retry_after_seconds=None):
    return SimpleNamespace(
        content=content,
        usage=usage or {"input_tokens": 0, "output_tokens": 0},
        error=error,
        retry_after_seconds=retry_after_seconds,
    )


def _text(text):
    return _FakeTextBlock(text)


def _tool_use(*, id, name, input):
    return _FakeToolUseBlock(id=id, name=name, input=input)


def _thinking(text="..."):
    return _FakeThinkingBlock(thinking=text)


def _result(subtype="success"):
    return SimpleNamespace(subtype=subtype)


def _stub_run_query(monkeypatch, handler):
    """Replace ``_run_query`` with an async fake driven by ``handler``.

    ``handler(prompt: str, tool_bridge) -> (first_assistant, last_result)``
    Also stubs ``_build_tool_bridge`` to a cheap sentinel so no
    scenario needs to touch the real MCP-tool wiring.
    """

    async def fake_run_query(*, model, prompt, system_prompt, tool_bridge, max_turns, effort=None):
        return handler(prompt, tool_bridge)

    monkeypatch.setattr(mod, "_run_query", fake_run_query)
    monkeypatch.setattr(
        mod, "_build_tool_bridge", lambda tools: ("fake-server", [f"mcp__openant__{t.name}" for t in tools])
    )


def _make_adapter() -> mod.ClaudeSubscriptionAdapter:
    return mod.ClaudeSubscriptionAdapter()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_constructs_when_sdk_importable(self):
        # The autouse fixture stubs the module; constructing must not raise.
        _make_adapter()

    def test_raises_auth_error_when_sdk_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
        # Force the import to fail regardless of what's actually installed
        # in the test environment.
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        with pytest.raises(LLMAuthError):
            mod.ClaudeSubscriptionAdapter()

    def test_satisfies_protocol(self):
        from utilities.llm import LLMAdapter

        adapter = _make_adapter()
        assert isinstance(adapter, LLMAdapter)
        assert adapter.name == "claude_subscription"
        assert adapter.supports_tools is True


# ---------------------------------------------------------------------------
# Effort configuration (OPENANT_CLAUDE_SUBSCRIPTION_EFFORT)
# ---------------------------------------------------------------------------


class TestEffortConfiguration:
    def test_unset_env_var_means_no_override(self, monkeypatch):
        monkeypatch.delenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", raising=False)
        adapter = mod.ClaudeSubscriptionAdapter()
        assert adapter._effort_override is None

    def test_valid_env_var_is_read(self, monkeypatch):
        monkeypatch.setenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", "medium")
        adapter = mod.ClaudeSubscriptionAdapter()
        assert adapter._effort_override == "medium"

    def test_invalid_env_var_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", "ludicrous")
        with pytest.raises(ValueError, match="ludicrous"):
            mod.ClaudeSubscriptionAdapter()

    def test_override_reaches_claude_agent_options(self, monkeypatch, _stub_claude_agent_sdk_module):
        """End-to-end: the env var override must actually reach the SDK's
        options object, not just get stored on the adapter."""
        monkeypatch.setenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", "low")
        sdk = _stub_claude_agent_sdk_module
        captured = {}

        async def fake_query(*, prompt, options):
            captured["effort"] = options.effort
            yield _FakeAssistantMessage(content=[_FakeTextBlock("hi")])
            yield _FakeResultMessage()

        sdk.query = fake_query

        adapter = mod.ClaudeSubscriptionAdapter()
        adapter.complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )

        assert captured["effort"] == "low"

    def test_no_signal_yet_reaches_options_as_none(self, monkeypatch, _stub_claude_agent_sdk_module):
        """No env var override and no observed RateLimitEvent yet: must
        reach the SDK as None (defer to its own "high" default) — not
        omitted, not a stale value from a prior test."""
        monkeypatch.delenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", raising=False)
        sdk = _stub_claude_agent_sdk_module
        captured = {}

        async def fake_query(*, prompt, options):
            captured["effort"] = options.effort
            yield _FakeAssistantMessage(content=[_FakeTextBlock("hi")])
            yield _FakeResultMessage()

        sdk.query = fake_query

        adapter = mod.ClaudeSubscriptionAdapter()
        adapter.complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )

        assert captured["effort"] is None


class TestDynamicEffortFromUtilization:
    """Effort is decided per-call from the most recently observed
    RateLimitEvent.rate_limit_info.utilization — not a static setting."""

    @pytest.mark.parametrize(
        "utilization,expected",
        [
            (None, None),
            (0.0, None),
            (0.49, None),
            (0.5, "medium"),
            (0.74, "medium"),
            (0.75, "low"),
            (0.99, "low"),
        ],
    )
    def test_mapping_thresholds(self, utilization, expected):
        assert mod._dynamic_effort_for_utilization(utilization) == expected

    def test_observed_rate_limit_event_lowers_effort_on_next_call(
        self, monkeypatch, _stub_claude_agent_sdk_module
    ):
        monkeypatch.delenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", raising=False)
        sdk = _stub_claude_agent_sdk_module
        captured = []

        async def fake_query(*, prompt, options):
            captured.append(options.effort)
            # First call: emit a high-utilization RateLimitEvent alongside
            # the answer, simulating the subscription reporting stress.
            if len(captured) == 1:
                yield _FakeRateLimitEvent(SimpleNamespace(
                    status="approaching_limit", utilization=0.8,
                    rate_limit_type="five_hour", resets_at=None,
                ))
            yield _FakeAssistantMessage(content=[_FakeTextBlock("hi")])
            yield _FakeResultMessage()

        sdk.query = fake_query
        adapter = mod.ClaudeSubscriptionAdapter()

        # First call: no signal observed yet -> None (SDK default).
        adapter.complete(
            model="claude-opus-4-6", system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )
        assert captured[0] is None

        # Second call: the RateLimitEvent from call 1 (utilization=0.8)
        # must have lowered this call's effort to "low".
        adapter.complete(
            model="claude-opus-4-6", system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )
        assert captured[1] == "low"

    def test_env_var_override_wins_over_observed_utilization(
        self, monkeypatch, _stub_claude_agent_sdk_module
    ):
        """An explicit pin must not be second-guessed by telemetry."""
        monkeypatch.setenv("OPENANT_CLAUDE_SUBSCRIPTION_EFFORT", "max")
        mod._UtilizationTracker.update(0.95)  # would otherwise force "low"

        sdk = _stub_claude_agent_sdk_module
        captured = {}

        async def fake_query(*, prompt, options):
            captured["effort"] = options.effort
            yield _FakeAssistantMessage(content=[_FakeTextBlock("hi")])
            yield _FakeResultMessage()

        sdk.query = fake_query
        adapter = mod.ClaudeSubscriptionAdapter()
        adapter.complete(
            model="claude-opus-4-6", system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )
        assert captured["effort"] == "max"

    def test_reset_warnings_clears_tracker(self):
        mod._UtilizationTracker.update(0.9)
        assert mod._UtilizationTracker.current() == 0.9
        mod.reset_warnings()
        assert mod._UtilizationTracker.current() is None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTextCompletion:
    def test_plain_text(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (
                _assistant([_text("hi there")], usage={"input_tokens": 3, "output_tokens": 5}),
                _result(),
            ),
        )
        adapter = _make_adapter()
        result = adapter.complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hello")])],
            max_tokens=64,
        )
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "hi there"
        assert result.input_tokens == 3
        assert result.output_tokens == 5
        assert result.stop_reason == "end_turn"

    def test_pricing_is_always_zero(self):
        adapter = _make_adapter()
        assert adapter.pricing.get("claude-opus-4-6") == {"input": 0.0, "output": 0.0}

    def test_cache_usage_is_read_through(self, monkeypatch):
        """The SDK's usage dict carries cache_creation_input_tokens /
        cache_read_input_tokens even though this adapter never requests
        caching explicitly (see module docstring point 5) — must reach
        CompletionResult, not be dropped."""
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (
                _assistant(
                    [_text("hi there")],
                    usage={
                        "input_tokens": 2,
                        "output_tokens": 6,
                        "cache_creation_input_tokens": 23012,
                        "cache_read_input_tokens": 19662,
                    },
                ),
                _result(),
            ),
        )
        adapter = _make_adapter()
        result = adapter.complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hello")])],
            max_tokens=64,
        )
        assert result.cache_creation_input_tokens == 23012
        assert result.cache_read_input_tokens == 19662

    def test_cache_usage_defaults_to_zero_when_absent(self, monkeypatch):
        # Older/other SDK builds may omit these keys entirely; must not KeyError.
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (
                _assistant([_text("hi there")], usage={"input_tokens": 3, "output_tokens": 5}),
                _result(),
            ),
        )
        adapter = _make_adapter()
        result = adapter.complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hello")])],
            max_tokens=64,
        )
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert adapter.pricing.get("anything-at-all") == {"input": 0.0, "output": 0.0}


# ---------------------------------------------------------------------------
# Transcript flattening
# ---------------------------------------------------------------------------


class TestTranscriptFlattening:
    def test_history_and_no_tool_bridge_when_no_tools(self, monkeypatch):
        captured = {}

        def handler(prompt, bridge):
            captured["prompt"] = prompt
            captured["bridge"] = bridge
            return _assistant([_text("ok")]), _result()

        _stub_run_query(monkeypatch, handler)
        adapter = _make_adapter()
        adapter.complete(
            model="claude-opus-4-6",
            system="You are helpful.",
            messages=[
                Message(role="user", content=[TextBlock("first")]),
                Message(role="assistant", content=[TextBlock("first reply")]),
                Message(role="user", content=[TextBlock("second")]),
            ],
            max_tokens=64,
        )
        assert "Human: first" in captured["prompt"]
        assert "Assistant: first reply" in captured["prompt"]
        assert "Human: second" in captured["prompt"]
        assert captured["bridge"] is None


# ---------------------------------------------------------------------------
# Thinking-only message skip (regression: caught via live testing against
# the real claude CLI — it yields a separate, thinking-only AssistantMessage
# BEFORE the one carrying the actual tool_use/text content)
# ---------------------------------------------------------------------------


class TestSubstantiveContentFilter:
    def test_thinking_only_is_not_substantive(self):
        msg = SimpleNamespace(content=[_thinking()])
        assert mod._has_substantive_content(msg) is False

    def test_tool_use_alongside_thinking_is_substantive(self):
        msg = SimpleNamespace(content=[_thinking(), _tool_use(id="t1", name="x", input={})])
        assert mod._has_substantive_content(msg) is True

    def test_text_is_substantive(self):
        msg = SimpleNamespace(content=[_text("hi")])
        assert mod._has_substantive_content(msg) is True


# ---------------------------------------------------------------------------
# Salvage a captured message when the SDK raises mid-stream (regression:
# hit live — the real subscription rate limit fired mid-test, the SDK
# yielded an AssistantMessage(error="rate_limit") and THEN raised a
# confusing internal exception while finishing the generator)
# ---------------------------------------------------------------------------


class TestRunQueryExceptionSalvage:
    def test_salvages_message_seen_before_a_mid_stream_exception(self, monkeypatch, _stub_claude_agent_sdk_module):
        sdk = _stub_claude_agent_sdk_module
        # Shape matches what was actually observed live: the rate-limited
        # message carries explanatory text, not empty content.
        rate_limited = _FakeAssistantMessage(content=[_FakeTextBlock("You've hit your session limit")])
        rate_limited.error = "rate_limit"

        async def fake_query(*, prompt, options):
            yield rate_limited
            raise RuntimeError("Claude Code returned an error result: success")

        sdk.query = fake_query
        import asyncio

        first_assistant, last_result = asyncio.run(
            mod._run_query(
                model="claude-opus-4-6",
                prompt="hi",
                system_prompt="",
                tool_bridge=None,
                max_turns=1,
            )
        )
        assert first_assistant is rate_limited
        assert last_result is None

    def test_reraises_when_nothing_was_captured(self, monkeypatch, _stub_claude_agent_sdk_module):
        sdk = _stub_claude_agent_sdk_module

        async def fake_query(*, prompt, options):
            raise RuntimeError("boom before anything arrived")
            yield  # pragma: no cover - makes this an async generator

        sdk.query = fake_query
        import asyncio

        with pytest.raises(RuntimeError):
            asyncio.run(
                mod._run_query(
                    model="claude-opus-4-6",
                    prompt="hi",
                    system_prompt="",
                    tool_bridge=None,
                    max_turns=1,
                )
            )


# ---------------------------------------------------------------------------
# Tool use round trip
# ---------------------------------------------------------------------------


class TestToolUseRoundTrip:
    def test_tool_use_then_end_turn(self, monkeypatch):
        def handler(prompt, bridge):
            assert bridge is not None
            if "Assistant:" not in prompt:
                return (
                    _assistant(
                        [_tool_use(id="toolu_1", name="echo", input={"text": "hello"})],
                        usage={"input_tokens": 10, "output_tokens": 8},
                    ),
                    _result(),
                )
            return (
                _assistant([_text("echoed: hello")], usage={"input_tokens": 20, "output_tokens": 4}),
                _result(),
            )

        _stub_run_query(monkeypatch, handler)
        adapter = _make_adapter()
        tools = [
            ToolDef(
                name="echo",
                description="Echo input",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            )
        ]

        first = adapter.complete(
            model="claude-opus-4-6",
            system="You are helpful.",
            messages=[Message(role="user", content=[TextBlock("call echo")])],
            max_tokens=64,
            tools=tools,
        )
        assert first.stop_reason == "tool_use"
        tool_uses = [b for b in first.content if isinstance(b, ToolUseBlock)]
        assert len(tool_uses) == 1
        tu = tool_uses[0]
        assert tu.name == "echo"
        assert tu.input == {"text": "hello"}

        second = adapter.complete(
            model="claude-opus-4-6",
            system="You are helpful.",
            messages=[
                Message(role="user", content=[TextBlock("call echo")]),
                Message(role="assistant", content=list(first.content)),
                Message(role="user", content=[ToolResultBlock(tool_use_id=tu.id, content='"hello"')]),
            ],
            max_tokens=64,
            tools=tools,
        )
        assert second.stop_reason == "end_turn"
        assert any(isinstance(b, TextBlock) for b in second.content)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_auth_error(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (_assistant([], error="authentication_failed"), None),
        )
        with pytest.raises(LLMAuthError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_billing_error_maps_to_auth_error(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (_assistant([], error="billing_error"), None),
        )
        with pytest.raises(LLMAuthError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_rate_limit_reports_to_global_limiter_and_carries_hint(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (
                _assistant([], error="rate_limit", retry_after_seconds=7),
                None,
            ),
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )
        assert exc_info.value.retry_after == 7
        assert get_rate_limiter("claude_subscription").is_in_backoff()

    def test_server_error_treated_as_rate_limit(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (_assistant([], error="server_error"), None),
        )
        with pytest.raises(LLMRateLimitError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_invalid_request_maps_to_response_error(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (_assistant([], error="invalid_request"), None),
        )
        with pytest.raises(LLMResponseError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_unexpected_exception_maps_to_connection_error(self, monkeypatch):
        async def boom(**kwargs):
            raise RuntimeError("subprocess spawn failed")

        monkeypatch.setattr(mod, "_run_query", boom)
        with pytest.raises(LLMConnectionError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_empty_completion_is_response_error(self, monkeypatch):
        _stub_run_query(monkeypatch, lambda prompt, bridge: (_assistant([]), _result()))
        with pytest.raises(LLMResponseError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )

    def test_no_assistant_message_at_all_is_response_error(self, monkeypatch):
        _stub_run_query(monkeypatch, lambda prompt, bridge: (None, _result(subtype="error_during_execution")))
        with pytest.raises(LLMResponseError):
            _make_adapter().complete(
                model="claude-opus-4-6",
                system=None,
                messages=[Message(role="user", content=[TextBlock("hi")])],
                max_tokens=8,
            )


# ---------------------------------------------------------------------------
# max_output_tokens — unique to this adapter
# ---------------------------------------------------------------------------


class TestMaxOutputTokens:
    def test_max_output_tokens_is_not_an_exception(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (
                _assistant([_text("truncated...")], error="max_output_tokens"),
                _result(),
            ),
        )
        result = _make_adapter().complete(
            model="claude-opus-4-6",
            system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=8,
        )
        assert result.stop_reason == "max_tokens"
        assert result.content[0].text == "truncated..."


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_ok(self, monkeypatch):
        _stub_run_query(monkeypatch, lambda prompt, bridge: (_assistant([_text("hi")]), _result()))
        assert _make_adapter().validate(model="claude-opus-4-6") is None

    def test_validate_auth_failure(self, monkeypatch):
        _stub_run_query(
            monkeypatch,
            lambda prompt, bridge: (_assistant([], error="authentication_failed"), None),
        )
        with pytest.raises(LLMAuthError):
            _make_adapter().validate(model="claude-opus-4-6")
