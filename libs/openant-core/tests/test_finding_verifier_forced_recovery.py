"""Tests for FindingVerifier's forced-final-verdict recovery
(_force_final_verdict) — the fix so exhausting the verify loop's
exploration budget (max iterations, a truncated turn, unparseable text)
doesn't by itself mean a vulnerability goes unassessed.

Each test's stub adapter behaves differently on its LAST call (the
forced-recovery attempt, restricted to the `finish` tool only) versus
its earlier calls — this is what distinguishes "recovery actually
works" from the existing degenerate-path tests, which only ever
confirm graceful bounded FAILURE.
"""

from __future__ import annotations

import json

import pytest

from utilities.agentic_enhancer.repository_index import RepositoryIndex
from utilities.finding_verifier import MAX_ITERATIONS, FindingVerifier
from utilities.llm import PhaseBinding, TextBlock, ToolUseBlock
from utilities.llm.adapter import CompletionResult
from utilities.llm_client import reset_warning_state

STAGE1_FINDING = "vulnerable"


@pytest.fixture(autouse=True)
def _reset():
    reset_warning_state()
    yield
    reset_warning_state()


def _make_verifier(adapter) -> FindingVerifier:
    binding = PhaseBinding(phase="verify", adapter=adapter, model="claude-x", provider_name="anthropic")
    return FindingVerifier(index=RepositoryIndex({}, repo_path=None), binding=binding)


def _verify(adapter):
    return _make_verifier(adapter).verify_result(
        code="x = 1", finding=STAGE1_FINDING, attack_vector="a", reasoning="r"
    )


def _finish_block(*, agree: bool, correct_finding: str = "vulnerable", explanation: str = "confirmed on recovery"):
    return ToolUseBlock(
        id="finish-recovery",
        name="finish",
        input={"agree": agree, "correct_finding": correct_finding, "explanation": explanation},
    )


class _NoToolCallsThenRecoversAdapter:
    """First call: truncated, no tool call (path #2). Recovery call:
    a real `finish` verdict."""

    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self):
        self.calls = 0
        self.tools_seen: list = []

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls += 1
        self.tools_seen.append(tools)
        if self.calls == 1:
            return CompletionResult(
                content=[TextBlock("partial reasoning that got cut off")],
                input_tokens=1, output_tokens=1, stop_reason="max_tokens",
            )
        return CompletionResult(
            content=[_finish_block(agree=True, correct_finding="safe", explanation="actually safe on closer look")],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
        )


class _UnparseableTextThenRecoversAdapter:
    """First call: end_turn with no JSON (path #1). Recovery call: a
    real `finish` verdict confirming the vulnerability."""

    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self):
        self.calls = 0

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                content=[TextBlock("hmm, let me think about this without calling any tool")],
                input_tokens=1, output_tokens=1, stop_reason="end_turn",
            )
        return CompletionResult(
            content=[_finish_block(agree=True, correct_finding="vulnerable")],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
        )


class _MaxIterationsThenRecoversAdapter:
    """Keeps calling a non-finish tool for MAX_ITERATIONS rounds, then
    the forced-recovery call (restricted to `finish` only) provides a
    real verdict."""

    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self):
        self.calls = 0
        self.tools_seen: list = []

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls += 1
        self.tools_seen.append(tools)
        if self.calls <= MAX_ITERATIONS:
            return CompletionResult(
                content=[ToolUseBlock(id=f"t{self.calls}", name="search_usages", input={"function_name": "noop"})],
                input_tokens=1, output_tokens=1, stop_reason="tool_use",
            )
        return CompletionResult(
            content=[_finish_block(agree=False, correct_finding="vulnerable", explanation="confirmed after budget exhausted")],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
        )


class _TextRecoveryAdapter:
    """Recovery call responds with parseable JSON text instead of a
    `finish` tool call — the _try_parse_text_response fallback path."""

    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self):
        self.calls = 0

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                content=[TextBlock("no json here")],
                input_tokens=1, output_tokens=1, stop_reason="end_turn",
            )
        payload = {"agree": True, "correct_finding": "vulnerable", "explanation": "recovered via text"}
        return CompletionResult(
            content=[TextBlock(json.dumps(payload))],
            input_tokens=1, output_tokens=1, stop_reason="end_turn",
        )


class TestForcedRecoverySucceeds:
    def test_recovers_a_real_verdict_after_no_tool_calls(self):
        adapter = _NoToolCallsThenRecoversAdapter()
        result = _verify(adapter)
        assert adapter.calls == 2, "exactly one bounded recovery call, not a loop"
        assert result.incomplete is False, "a real recovered verdict must not read as 'unverified'"
        assert result.agree is True
        assert result.correct_finding == "safe"
        assert result.explanation == "actually safe on closer look"

    def test_recovers_a_real_verdict_after_unparseable_text(self):
        adapter = _UnparseableTextThenRecoversAdapter()
        result = _verify(adapter)
        # 3, not 2: an end_turn with unparseable text ALSO triggers the
        # pre-existing JSONCorrector fallback (_parse_json_from_text)
        # before _force_final_verdict ever runs — that correction
        # attempt is call 2 and fails harmlessly (the stub's canned
        # finish-tool response isn't the plain-text JSON JSONCorrector
        # expects), then the actual forced-recovery call (call 3)
        # succeeds. Still fully bounded, no loop.
        assert adapter.calls == 3
        assert result.incomplete is False
        assert result.agree is True
        assert result.correct_finding == "vulnerable"

    def test_recovers_a_real_verdict_after_max_iterations(self):
        adapter = _MaxIterationsThenRecoversAdapter()
        result = _verify(adapter)
        assert adapter.calls == MAX_ITERATIONS + 1, "bounded: MAX_ITERATIONS + exactly one recovery call"
        assert result.incomplete is False, (
            "this is the exact scenario the fix targets: don't miss a real "
            "vulnerability just because the exploration budget ran out"
        )
        assert result.correct_finding == "vulnerable"
        assert "confirmed after budget exhausted" in result.explanation

    def test_recovery_call_restricts_tools_to_finish_only(self):
        adapter = _MaxIterationsThenRecoversAdapter()
        _verify(adapter)
        recovery_tools = adapter.tools_seen[-1]
        assert [t.name for t in recovery_tools] == ["finish"], (
            "the recovery call must not be able to reopen another "
            "exploration round — only `finish` should be offered"
        )

    def test_recovers_via_parseable_text_response_too(self):
        adapter = _TextRecoveryAdapter()
        result = _verify(adapter)
        # 3, not 2 — same JSONCorrector-fires-first reason as the
        # unparseable-text recovery test above.
        assert adapter.calls == 3
        assert result.incomplete is False
        assert result.agree is True
        assert result.correct_finding == "vulnerable"
        assert result.explanation == "recovered via text"


class TestForcedRecoveryDoesNotBreakGenuineDisagreement:
    def test_a_real_disagree_verdict_from_recovery_is_honored(self):
        # Recovery producing agree=False with a "safe" correct_finding is a
        # REAL, completed verdict (incomplete=False) — distinct from the
        # incomplete=True fail-safe paths, which always force agree=False
        # AND preserve the Stage-1 finding. A genuine recovered verdict
        # must be trusted as-is, including downgrading to "safe".
        class _Adapter:
            name = "anthropic"
            supports_tools = True
            pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

            def __init__(self):
                self.calls = 0

            def complete(self, *, model, system, messages, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return CompletionResult(
                        content=[TextBlock("thinking...")],
                        input_tokens=1, output_tokens=1, stop_reason="end_turn",
                    )
                return CompletionResult(
                    content=[_finish_block(agree=False, correct_finding="safe", explanation="not exploitable after all")],
                    input_tokens=1, output_tokens=1, stop_reason="tool_use",
                )

        result = _verify(_Adapter())
        assert result.incomplete is False
        assert result.agree is False
        assert result.correct_finding == "safe"


class TestForcedRecoveryFailureStillBounded:
    def test_recovery_calls_that_always_raise_fall_back_cleanly(self):
        class _AlwaysRaisesOnRecovery:
            name = "anthropic"
            supports_tools = True
            pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

            def __init__(self):
                self.calls = 0

            def complete(self, *, model, system, messages, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return CompletionResult(
                        content=[TextBlock("partial reasoning that got cut off")],
                        input_tokens=1, output_tokens=1, stop_reason="max_tokens",
                    )
                raise RuntimeError("network blip during recovery attempt")

        adapter = _AlwaysRaisesOnRecovery()
        result = _verify(adapter)  # must not raise/propagate
        # 1 normal call + MAX_RECOVERY_ATTEMPTS (3) recovery attempts,
        # every one of which raises — still terminates, doesn't retry
        # forever.
        from utilities.finding_verifier import MAX_RECOVERY_ATTEMPTS
        assert adapter.calls == 1 + MAX_RECOVERY_ATTEMPTS
        assert result.incomplete is True
        assert result.agree is False
        assert result.correct_finding == STAGE1_FINDING


class TestForcedRecoveryRetriesAcrossAttempts:
    def test_succeeds_on_a_later_attempt_after_earlier_ones_fail(self):
        # This is the direct proof of the multi-attempt widening: the
        # first recovery attempt errors (simulating one pool member
        # having an off moment), the second still doesn't produce a
        # verdict, and the THIRD finally succeeds — exactly the
        # "don't give up after one bad attempt" behavior requested.
        class _SucceedsOnThirdRecoveryAttempt:
            name = "anthropic"
            supports_tools = True
            pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

            def __init__(self):
                self.calls = 0

            def complete(self, *, model, system, messages, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return CompletionResult(
                        content=[TextBlock("partial reasoning that got cut off")],
                        input_tokens=1, output_tokens=1, stop_reason="max_tokens",
                    )
                if self.calls == 2:
                    raise RuntimeError("transient error on first recovery attempt")
                if self.calls == 3:
                    return CompletionResult(
                        content=[TextBlock("still nothing useful")],
                        input_tokens=1, output_tokens=1, stop_reason="end_turn",
                    )
                return CompletionResult(
                    content=[_finish_block(agree=True, correct_finding="vulnerable", explanation="confirmed on 3rd try")],
                    input_tokens=1, output_tokens=1, stop_reason="tool_use",
                )

        adapter = _SucceedsOnThirdRecoveryAttempt()
        result = _verify(adapter)
        assert result.incomplete is False, "a later attempt succeeding must still count as a real, completed verdict"
        assert result.agree is True
        assert result.correct_finding == "vulnerable"
        assert result.explanation == "confirmed on 3rd try"
