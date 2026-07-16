"""Stage 1 and Stage 2 prompts must cap inlined code, not send it unbounded.

The agentic enhancer (agentic_enhancer/agent.py) inlines dependency
functions into a unit's primary_code with no size limit — a unit with many
included dependencies can grow the code sent to Claude arbitrarily large.
Both Stage 1 (vulnerability_analysis.get_analysis_prompt) and Stage 2
(verification_prompts.get_verification_prompt) consume that same string, so
each caps it at the point it's interpolated into the prompt (same pattern
already used for the enhance loop's own input, see agentic_enhancer/prompts.py).

These tests are model-free pure string assertions — no LLM calls.
"""

from __future__ import annotations

from prompts._fence import cap_code
from prompts.verification_prompts import get_verification_prompt
from prompts.vulnerability_analysis import get_analysis_prompt
from utilities.context_enhancer import get_context_enhancement_prompt


class TestCapCode:
    def test_short_text_untouched(self):
        assert cap_code("short") == "short"

    def test_long_text_truncated_with_marker(self):
        text = "x" * 100
        capped = cap_code(text, limit=20)
        assert len(capped) == 20
        assert capped.endswith("(truncated)")

    def test_none_and_empty_tolerated(self):
        assert cap_code(None) == ""
        assert cap_code("") == ""

    def test_exactly_at_limit_untouched(self):
        text = "x" * 20
        assert cap_code(text, limit=20) == text


class TestAnalysisPromptCapsCode:
    def test_oversized_single_function_capped(self):
        code = "x" * 200_000
        prompt = get_analysis_prompt(code=code, language="python")
        assert "x" * 200_000 not in prompt
        assert "(truncated)" in prompt

    def test_oversized_context_section_capped_independently_of_primary(self):
        primary = "def target(): pass"
        huge_context = "y" * 200_000
        code = primary + "\n// ========== File Boundary ==========\n" + huge_context
        prompt = get_analysis_prompt(code=code, language="python")
        # The target function must survive uncapped — only the
        # supplementary context is expendable.
        assert primary in prompt
        assert "y" * 200_000 not in prompt
        assert "(truncated)" in prompt


class TestContextEnhancementPromptCapsCode:
    # Single-shot (non-agentic) enhance mode — utilities/context_enhancer.py.
    # Separate call site, separate cap enforcement point from Stage 1/2.
    def test_oversized_function_code_capped(self):
        prompt = get_context_enhancement_prompt(
            function_id="a.js:fn",
            function_name="fn",
            function_code="x" * 200_000,
            unit_type="function",
            class_name=None,
            static_deps=[],
            static_callers=[],
            context_functions=[],
        )
        assert "x" * 200_000 not in prompt
        assert "(truncated)" in prompt

    def test_short_function_code_untouched(self):
        prompt = get_context_enhancement_prompt(
            function_id="a.js:fn",
            function_name="fn",
            function_code="function fn() { return 1; }",
            unit_type="function",
            class_name=None,
            static_deps=[],
            static_callers=[],
            context_functions=[],
        )
        assert "function fn() { return 1; }" in prompt
        assert "(truncated)" not in prompt


class TestVerificationPromptCapsCode:
    def test_oversized_single_function_capped(self):
        code = "x" * 200_000
        prompt = get_verification_prompt(
            code=code, finding="vulnerable", attack_vector="x", reasoning="y"
        )
        assert "x" * 200_000 not in prompt
        assert "(truncated)" in prompt

    def test_oversized_context_section_capped_independently_of_primary(self):
        primary = "def target(): pass"
        huge_context = "y" * 200_000
        code = primary + "\n// ========== File Boundary ==========\n" + huge_context
        prompt = get_verification_prompt(
            code=code, finding="vulnerable", attack_vector="x", reasoning="y"
        )
        assert primary in prompt
        assert "y" * 200_000 not in prompt
        assert "(truncated)" in prompt
