"""Communication-style directives in the two agentic (tool-calling) loops.

Claude models are documented to narrate more between tool calls in agentic
sessions than the actual analysis requires ("Let me check...", "Now I'll
look at..." before every tool call) — text that costs output tokens without
adding analytical value, unlike the model's reasoning that goes into tool
call arguments and the structured finish-tool fields. Both agentic loops
(enhance's ContextAgent, verify's FindingVerifier) got an explicit
communication-style instruction to cut that narration, scoped narrowly:
it targets prose *between* tool calls, not the depth of investigation or
the structured exploit_path/attack_steps/classification_reasoning content
the model is asked to report.

Stage 1 (analyze) is not covered here — it's a single non-agentic call
with a fixed JSON output shape, so there's no inter-tool-call narration to
reduce in the first place.

These are pure string-membership assertions on the prompt text — no LLM
calls involved.
"""

from __future__ import annotations

from prompts.verification_prompts import VERIFICATION_SYSTEM_PROMPT, get_verification_system_prompt
from utilities.agentic_enhancer.prompts import SYSTEM_PROMPT as ENHANCE_SYSTEM_PROMPT


class TestEnhanceLoopCommunicationStyle:
    def test_instructs_against_narrating_tool_calls(self):
        assert "Communication Style" in ENHANCE_SYSTEM_PROMPT
        assert "narrate" in ENHANCE_SYSTEM_PROMPT.lower()

    def test_points_analysis_at_classification_reasoning_field(self):
        # The cut must redirect substance into the structured output, not
        # just delete it.
        assert "classification_reasoning" in ENHANCE_SYSTEM_PROMPT


class TestVerifyLoopCommunicationStyle:
    def test_instructs_against_narrating_tool_calls(self):
        assert "Communication Style" in VERIFICATION_SYSTEM_PROMPT
        assert "narrate" in VERIFICATION_SYSTEM_PROMPT.lower()

    def test_points_exploit_trace_at_structured_fields(self):
        # The step-by-step exploit trace is substantive, not narration —
        # it must still be captured, just in the tool call, not as prose.
        assert "exploit_path" in VERIFICATION_SYSTEM_PROMPT

    def test_survives_app_context_suppression_append(self):
        # get_verification_system_prompt appends CLI-tool suppression text
        # when relevant; the communication-style instruction must still be
        # present in the combined result, not just the bare constant.
        prompt = get_verification_system_prompt(app_context=None)
        assert "Communication Style" in prompt
