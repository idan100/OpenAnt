"""Stage 1 must require PROOF of a tainted value's origin before a
"vulnerable" verdict, not just an assumption based on a variable's name
or a "looks user-facing" code pattern (path.join + fs op, etc).

Regression target: a CLI script's argument (process.argv-sourced)
flagged as remotely attacker-controlled path traversal purely because
the code SHAPE looked dangerous, with no proof the value crosses an
actual trust boundary. See STAGE1_SYSTEM_PROMPT / get_analysis_prompt
in prompts/vulnerability_analysis.py for the fix itself; this file
only guards that the required language stays present.
"""

from __future__ import annotations

from prompts.vulnerability_analysis import (
    STAGE1_SYSTEM_PROMPT,
    get_analysis_prompt,
    get_system_prompt,
)


class TestSystemPromptRequiresOriginProof:
    def test_system_prompt_requires_tracing_the_origin(self):
        assert "genuinely untrusted" in STAGE1_SYSTEM_PROMPT
        assert "trust boundary" in STAGE1_SYSTEM_PROMPT

    def test_system_prompt_names_cli_args_as_not_attacker_controlled_by_default(self):
        assert "CLI argument" in STAGE1_SYSTEM_PROMPT
        assert "NOT" in STAGE1_SYSTEM_PROMPT

    def test_system_prompt_instructs_defaulting_to_safe_when_unproven(self):
        assert "lean" in STAGE1_SYSTEM_PROMPT.lower()
        assert "inconclusive" in STAGE1_SYSTEM_PROMPT.lower()

    def test_get_system_prompt_includes_it_regardless_of_app_context(self):
        # No app_context (the exact scenario the reported false positive
        # occurred in — a repo with no app_context configured at all).
        assert "genuinely untrusted" in get_system_prompt(app_context=None)


class TestAnalysisPromptRequiresOriginProof:
    def test_input_question_without_app_context_requires_tracing(self):
        prompt = get_analysis_prompt(code="function f(x) {}", language="javascript", app_context=None)
        assert "EXACT origin" in prompt
        assert "Can't actually trace it" in prompt
        assert 'lean toward "safe" or "inconclusive"' in prompt

    def test_prove_it_step_requires_justifying_the_entry_point(self):
        prompt = get_analysis_prompt(code="function f(x) {}", language="javascript", app_context=None)
        assert "how do you know" in prompt
        assert "not just assumed from the" in prompt

    def test_reasoning_field_guidance_requires_stating_the_origin(self):
        prompt = get_analysis_prompt(code="function f(x) {}", language="javascript", app_context=None)
        assert "MUST state the specific traced origin" in prompt

    def test_default_to_safe_line_requires_proven_origin(self):
        prompt = get_analysis_prompt(code="function f(x) {}", language="javascript", app_context=None)
        assert "prove the tainted value's origin is genuinely attacker-controlled" in prompt
