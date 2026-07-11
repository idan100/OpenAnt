"""Claude Agent SDK adapter — rides a Claude Pro/Max subscription login.

Every other adapter in this package bills through a provider's metered
API (an ``api_key`` against ``api.anthropic.com`` / OpenAI / Google).
This one instead shells out — via the ``claude-agent-sdk`` Python
package — to a locally installed, already-authenticated ``claude`` CLI
(``claude login``). Completions then draw on the user's Claude Pro or
Max subscription instead of a separate, metered Anthropic API key.

Read this before touching the file:

1. **``claude-agent-sdk`` is an optional dependency, not a hard one.**
   It requires the Claude Code CLI (Node.js) to be installed and
   logged in separately — most OpenAnt users won't have that, so it's
   NOT in ``requirements.txt`` / ``pyproject.toml`` core deps. Install
   it with ``pip install claude-agent-sdk`` (extras group
   ``claude-subscription``). The import happens lazily in
   :meth:`ClaudeSubscriptionAdapter.__init__` so importing this
   *module* never fails — only *constructing* the adapter does, with
   a message pointing at what to install.

2. **The Agent SDK is not a raw completion API — it's Claude Code.**
   ``query()``/``ClaudeSDKClient`` run Claude Code's own, self-driving
   agent loop (it decides which tools to call and executes them
   itself). ``LLMAdapter.complete()`` needs the opposite: a single
   model turn, with any ``ToolUseBlock`` handed back to *OpenAnt's*
   pipeline to execute, not run internally. Bridging the two means:

   * Every ``complete()`` call opens a **fresh, stateless** ``query()``
     session — matching how OpenAnt already resends the full message
     history on every call (adapters are stateless dispatchers; see
     ``adapter.py``'s module docstring).
   * The conversation history OpenAnt maintains (``messages``) is
     flattened into one "Human: ... / Assistant: ..." transcript
     string and sent as the ``query()`` prompt, since ``query()``'s
     streaming-input mode is documented only for injecting new
     *user* turns into a live session, not for replaying an
     arbitrary multi-turn + tool-use history into a stateless one.
   * Tools OpenAnt passes are registered as in-process MCP tools
     (``@tool`` / ``create_sdk_mcp_server``). Their implementation
     does NOT run real logic — real tool execution stays in
     OpenAnt's pipeline. The stub just returns an ack so Claude
     Code's loop can close out the turn; OpenAnt only ever reads the
     **first** ``AssistantMessage`` (the ``ToolUseBlock`` it wanted)
     and discards anything the SDK generates after feeding back that
     stub result. ``max_turns`` bounds the wasted follow-up turn.
   * Claude Code's own built-in tools (Bash/Read/Write/...) are
     locked out (see ``_BUILTIN_TOOL_NAMES``) so a pipeline call never
     gets filesystem/shell access it didn't ask for.

3. **``max_tokens`` is advisory only.** The Agent SDK has no
   documented hard output-token cap (unlike the raw Messages API's
   ``max_tokens``) — Claude Code manages its own generation length.
   ``ponytail:`` known limitation, not enforced; upgrade path is to
   wire it through if/when the SDK exposes one.

4. **Every phase using this adapter draws on the human's own
   subscription quota** (not a service account) — running OpenAnt
   with this provider consumes the same Pro/Max usage the user would
   otherwise spend chatting with Claude. It's not "free" in the sense
   of having no cost, just not separately *billed*.

See ``utilities/llm/adapter.py`` for the protocol every adapter
implements, and ``providers/anthropic.py`` for the reference metered
adapter this one deliberately departs from.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any, Optional

from ._ratelimit import report_rate_limit, wait_for_rate_limit
from .._redact import redact_secrets, redacted_cause_from
from ..adapter import (
    CompletionResult,
    ContentBlock,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    Message,
    StopReason,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)


# Claude Code's own built-in tools. Explicitly denied on every call so a
# pipeline phase never gets incidental filesystem/shell/network access
# it didn't ask for — OpenAnt's tool-calling phases bring their own
# ToolDefs and expect ONLY those to be reachable. Not exhaustive by
# name-stability guarantee (Claude Code can add tools), which is why
# ``options.tools`` is ALSO set to an explicit allowlist (belt + braces
# — see ``_run_query``).
_BUILTIN_TOOL_NAMES = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "NotebookEdit", "Task",
]

# Message-level error codes the Agent SDK surfaces on
# ``AssistantMessage.error`` (see module docstring point 2 — this is
# the primary error-signalling path; the SDK doesn't document a typed
# Python exception hierarchy for these). "max_output_tokens" is handled
# separately in ``_translate`` (it's a normal stop condition, not an
# LLMError).
_ERROR_MESSAGES = {
    "authentication_failed": (
        LLMAuthError,
        "Claude Code CLI reports authentication_failed — run `claude login` "
        "to authenticate with your Claude Pro/Max subscription.",
    ),
    "billing_error": (
        LLMAuthError,
        "Claude Code CLI reports a billing_error — check your subscription "
        "status at claude.ai/settings/billing.",
    ),
    "rate_limit": (
        LLMRateLimitError,
        "Claude Code CLI reports rate_limit — the subscription's usage cap "
        "was likely reached; it resets on your plan's normal cycle.",
    ),
    "server_error": (
        LLMRateLimitError,
        "Claude Code CLI reports a server_error; treating it as transient "
        "for backoff purposes.",
    ),
    "invalid_request": (
        LLMResponseError,
        "Claude Code CLI reports invalid_request.",
    ),
    "unknown": (
        LLMResponseError,
        "Claude Code CLI reported an unknown error.",
    ),
}

# One-time-per-process warning bookkeeping for content-block kinds we
# receive but don't translate (mirrors AnthropicAdapter's pattern in
# providers/anthropic.py).
_warned_block_kinds: set[str] = set()
_warned_block_kinds_lock = threading.Lock()


def _warn_unknown_block_kind(kind: str) -> None:
    should_warn = False
    with _warned_block_kinds_lock:
        if kind not in _warned_block_kinds:
            _warned_block_kinds.add(kind)
            should_warn = True
    if should_warn:
        sys.stderr.write(
            f"warning: ClaudeSubscriptionAdapter received unknown content "
            f"block kind {kind!r}; dropping it.\n"
        )


class _ZeroCostPricing(dict):
    """Every lookup reports $0 — subscription billing has no marginal
    per-token cost, unlike the metered adapters. This is the CORRECT
    number (not a placeholder for "unknown"), so it deliberately skips
    the "no pricing for model X" warning that an empty ``{}`` would
    trigger in ``utilities/llm_client.py``.
    """

    def get(self, key: object, default: object = None) -> dict:  # noqa: ARG002
        return {"input": 0.0, "output": 0.0}


class ClaudeSubscriptionAdapter:
    """:class:`LLMAdapter` implementation backed by ``claude-agent-sdk``."""

    name = "claude_subscription"
    supports_tools = True
    pricing: dict = _ZeroCostPricing()

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,  # noqa: ARG002 - signature parity, unused
        base_url: Optional[str] = None,  # noqa: ARG002 - signature parity, unused
    ):
        # ``api_key``/``base_url`` are accepted (and ignored) so
        # ``registry.build_adapter`` — which always passes both — can
        # construct this adapter without a special case. There's no
        # key to check: auth comes from the local `claude login`
        # session the ``claude`` CLI reads at subprocess-spawn time.
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise LLMAuthError(
                "claude-agent-sdk is not installed. Install it with "
                "`pip install claude-agent-sdk`, install the Claude Code "
                "CLI (see https://code.claude.com/docs/en/agent-sdk for "
                "current instructions), and run `claude login` once to "
                "authenticate with your Claude Pro/Max subscription."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: Optional[str],
        messages: list[Message],
        max_tokens: int,  # noqa: ARG002 - advisory only, see module docstring point 3
        tools: Optional[list[ToolDef]] = None,
    ) -> CompletionResult:
        wait_for_rate_limit()

        prompt = _flatten_transcript(messages)
        system_prompt = system if system is not None else ""

        tool_bridge = _build_tool_bridge(tools) if tools else None
        # 1 turn when there's nothing to call (plain completion). With
        # tools, allow a 2nd turn as slack: our tool stub returns an ack
        # so Claude Code's loop can close turn 1 cleanly before hitting
        # the cap, rather than the cap landing mid-tool-dispatch. Either
        # way only the FIRST AssistantMessage is ever used — see module
        # docstring point 2.
        max_turns = 2 if tool_bridge else 1

        try:
            first_assistant, last_result = asyncio.run(
                _run_query(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    tool_bridge=tool_bridge,
                    max_turns=max_turns,
                )
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK boundary, map to typed error
            raise LLMConnectionError(
                redact_secrets(f"claude-agent-sdk error: {exc}")
            ) from redacted_cause_from(exc)

        try:
            return _translate(first_assistant, last_result)
        except LLMRateLimitError as exc:
            report_rate_limit(exc.retry_after)
            raise

    def validate(self, model: str) -> None:
        # Cheapest possible probe: one plain-text turn, no tools.
        # Reused via complete() rather than duplicated — this adapter
        # has no lower-level "just hit the endpoint" call the way the
        # raw-SDK adapters do (there's no endpoint to hit directly,
        # only the CLI subprocess), so there's nothing complete() does
        # that a validate()-specific path would skip.
        self.complete(
            model=model,
            system=None,
            messages=[Message(role="user", content=[TextBlock("hi")])],
            max_tokens=1,
        )


# ----------------------------------------------------------------------
# Transcript flattening
# ----------------------------------------------------------------------


def _flatten_transcript(messages: list[Message]) -> str:
    """Render OpenAnt's structured message history as a single prompt.

    ``query()`` accepts a plain string as one fresh user turn — there's
    no documented way to seed a stateless session with prior
    assistant/tool-use turns directly (see module docstring point 2).
    So multi-turn history is rendered as a "Human: ... / Assistant: ..."
    transcript and handed over as ONE user message, with an explicit
    instruction telling the model where to continue.

    The common case — a single fresh user turn, i.e. every phase's
    FIRST call — is passed through completely unwrapped instead.
    Measured live against the real CLI: wrapping a single-turn prompt
    in "Human: ... / Continue as Assistant..." framing measurably made
    the model answer in plain text instead of calling an available
    tool, even when explicitly asked to. The framing is only needed
    (and only added) once there's actual multi-turn history to
    represent — nothing to disambiguate on turn one.
    """
    if len(messages) == 1 and messages[0].role == "user":
        parts = [b.text for b in messages[0].content if isinstance(b, TextBlock)]
        if len(parts) == len(messages[0].content):  # no ToolUseBlock/ToolResultBlock
            return "\n".join(parts)

    lines: list[str] = []
    for message in messages:
        speaker = "Human" if message.role == "user" else "Assistant"
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(
                    f"[called tool {block.name!r} with input "
                    f"{json.dumps(block.input)}]"
                )
            elif isinstance(block, ToolResultBlock):
                label = block.name or block.tool_use_id
                parts.append(f"[result of tool {label!r}: {block.content}]")
        lines.append(f"{speaker}: {chr(10).join(parts)}")

    lines.append(
        "Continue the conversation as Assistant. Respond with only your "
        "next Assistant turn — do not repeat prior turns."
    )
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# Tool bridge (see module docstring point 2)
# ----------------------------------------------------------------------


async def _tool_stub(_args: dict[str, Any]) -> dict[str, Any]:
    """Filler MCP-tool implementation.

    Never performs the tool's real work — OpenAnt's pipeline does that
    itself once ``complete()`` returns the captured ``ToolUseBlock``.
    This only exists so Claude Code's internal loop has SOMETHING to
    feed back to the model and can close the turn cleanly.
    """
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    "Recorded. The caller will supply the real result in "
                    "a follow-up request."
                ),
            }
        ]
    }


def _build_tool_bridge(tools: list[ToolDef]):
    """Register ``tools`` as in-process MCP tools; return (server, allowlist)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    sdk_tools = [
        sdk_tool(name=t.name, description=t.description, input_schema=t.input_schema)(
            _tool_stub
        )
        for t in tools
    ]
    server = create_sdk_mcp_server(name="openant", version="1.0.0", tools=sdk_tools)
    allowlist = [f"mcp__openant__{t.name}" for t in tools]
    return server, allowlist


# ----------------------------------------------------------------------
# The query() call
# ----------------------------------------------------------------------


async def _run_query(
    *,
    model: str,
    prompt: str,
    system_prompt: str,
    tool_bridge,
    max_turns: int,
):
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

    mcp_servers: dict[str, Any] = {}
    tools_allowlist: list[str] = []
    if tool_bridge is not None:
        server, allowlist = tool_bridge
        mcp_servers = {"openant": server}
        tools_allowlist = allowlist

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        # Explicit allowlist (not the "claude_code" preset) — an empty
        # list disables every built-in tool. Combined with
        # disallowed_tools below as belt-and-braces (see
        # _BUILTIN_TOOL_NAMES).
        tools=tools_allowlist,
        allowed_tools=tools_allowlist,
        disallowed_tools=list(_BUILTIN_TOOL_NAMES),
        # Safe because the only tools reachable are our own in-process
        # stubs (no filesystem/shell/network access) — bypassing the
        # permission prompt avoids a hang in this non-interactive
        # pipeline, which has no TTY to answer it.
        permission_mode="bypassPermissions",
        mcp_servers=mcp_servers,
        max_turns=max_turns,
        # Load NONE of the user's personal ~/.claude / project settings
        # (hooks, plugins, slash commands). Without this, the subprocess
        # inherits whatever the human has installed for their own
        # interactive sessions — e.g. a SessionStart hook that injects
        # thousands of unrelated tokens into the system prompt on every
        # single OpenAnt call, observed live during development. A
        # deterministic security-scanning pipeline must not depend on
        # what plugins happen to be installed on the machine running it.
        setting_sources=[],
    )

    first_assistant = None
    last_result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            # The SDK can yield a separate, thinking-only AssistantMessage
            # BEFORE the one carrying the actual text/tool_use content
            # (observed live: one message with only a ThinkingBlock, then
            # a second with the ToolUseBlock). Skip past those — capturing
            # the thinking-only one as `first_assistant` would leave
            # `_translate` with nothing but a dropped ThinkingBlock and a
            # false "empty completion" error.
            if first_assistant is None and _has_substantive_content(message):
                first_assistant = message
        elif isinstance(message, ResultMessage):
            last_result = message

    return first_assistant, last_result


def _has_substantive_content(message) -> bool:
    # The Agent SDK's content-block dataclasses (TextBlock, ToolUseBlock,
    # ThinkingBlock, ...) carry NO ``.type`` discriminator field — unlike
    # the raw Anthropic SDK's response objects, they're distinguished by
    # Python class via isinstance(). Confirmed against the installed
    # package: ``dataclasses.fields(ThinkingBlock)`` is only
    # ``('thinking', 'signature')``.
    from claude_agent_sdk import ThinkingBlock as SdkThinkingBlock

    return any(not isinstance(block, SdkThinkingBlock) for block in message.content)


# ----------------------------------------------------------------------
# Result translation
# ----------------------------------------------------------------------


def _translate(first_assistant, last_result) -> CompletionResult:
    if first_assistant is None:
        subtype = getattr(last_result, "subtype", "unknown")
        raise LLMResponseError(
            f"claude CLI session ended without producing a response "
            f"(subtype={subtype!r}); it may have hit an internal turn/budget "
            f"limit before generating anything."
        )

    error_code = getattr(first_assistant, "error", None)
    if error_code and error_code != "max_output_tokens" and error_code in _ERROR_MESSAGES:
        error_cls, message = _ERROR_MESSAGES[error_code]
        if error_cls is LLMRateLimitError:
            # The Agent SDK's error literal carries no retry-hint field
            # today (unlike a raw 429's retry-after header). Read one
            # opportunistically in case a future SDK version — or a
            # test double — attaches it; ``getattr`` is safe against a
            # real AssistantMessage that never has this attribute.
            retry_after = getattr(first_assistant, "retry_after_seconds", None)
            raise LLMRateLimitError(message, retry_after=retry_after)
        raise error_cls(message)

    # Dispatch by isinstance, not a ``.type`` field — see
    # ``_has_substantive_content`` for why. Aliased to avoid shadowing
    # this module's own ``TextBlock``/``ToolUseBlock`` (our adapter
    # contract's types, imported from ``..adapter``).
    from claude_agent_sdk import TextBlock as SdkTextBlock
    from claude_agent_sdk import ThinkingBlock as SdkThinkingBlock
    from claude_agent_sdk import ToolUseBlock as SdkToolUseBlock

    content_blocks: list[ContentBlock] = []
    for block in first_assistant.content:
        if isinstance(block, SdkTextBlock):
            content_blocks.append(TextBlock(text=block.text))
        elif isinstance(block, SdkToolUseBlock):
            content_blocks.append(
                ToolUseBlock(id=block.id, name=block.name, input=block.input or {})
            )
        elif isinstance(block, SdkThinkingBlock):
            continue  # not part of our contract; dropped silently
        else:
            _warn_unknown_block_kind(type(block).__name__)

    if not content_blocks:
        raise LLMResponseError(
            "claude CLI returned no usable content (empty completion); the "
            "request may have been filtered or the response was malformed"
        )

    has_tool_use = any(isinstance(b, ToolUseBlock) for b in content_blocks)
    stop_reason: StopReason
    if error_code == "max_output_tokens":
        stop_reason = "max_tokens"
    elif has_tool_use:
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    usage = getattr(first_assistant, "usage", None) or {}
    return CompletionResult(
        content=content_blocks,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        stop_reason=stop_reason,
        raw=first_assistant,
    )
