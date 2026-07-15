"""Shared Markdown code-fence helper for prompt builders.

Both the Stage-1 analysis prompt (`vulnerability_analysis.py`) and the Stage-2
verification prompt (`verification_prompts.py`) interpolate UNTRUSTED analyzed
source code into Markdown code fences. Per the CommonMark spec, a fenced code
block opened with N backticks is closed by the first subsequent line that is a
run of >= N backticks. A bare ``` fence is therefore escapable: untrusted
content containing its own ``` line breaks out of the fence and the remainder is
read as prompt-level instructions (prompt injection — the attacker can steer the
analyst's / verifier's verdict).

This module centralises the one safe-fence implementation so both prompt
builders share identical behaviour (no duplication).
"""

from __future__ import annotations

import re


def cap_code(text: str, limit: int = 60_000) -> str:
    """Truncate ``text`` to ``limit`` chars with an explicit truncation marker.

    Analyzed source code reaches these prompts with no inherent size bound:
    the agentic enhancer (``agentic_enhancer/agent.py``) can inline an
    unbounded number of dependency functions into a unit's ``primary_code``,
    and that same blob is then sent to both Stage 1 (analysis) and Stage 2
    (verification, one call per unit, up to ``MAX_ITERATIONS`` times).
    ~4 chars/token, so 60k chars stays well under the model's context window
    per call. Truncating (rather than erroring) keeps the pipeline running on
    oversized units; the explicit marker stops the model from reasoning over
    partial code as if it were complete.
    """
    if not text or len(text) <= limit:
        return text or ""
    marker = "\n... (truncated)"
    return text[: limit - len(marker)] + marker


def safe_code_fence(text: str) -> str:
    """Return a backtick run guaranteed to enclose ``text`` un-escapably.

    The returned run is STRICTLY LONGER than the longest consecutive backtick
    run anywhere in ``text`` (minimum 3). No line inside the content can then
    satisfy the CommonMark closing rule (a line of >= N backticks), so the
    content stays inert data and cannot break out to inject prompt-level
    instructions.

    Callers that need a language info-string open with ``safe_code_fence(text)
    + language`` and close with the bare ``safe_code_fence(text)`` — both share
    this same length-aware run so the content cannot close the block early.
    """
    # Defensive: tolerate a None/empty body (a missing context block, an
    # empty unit) rather than raising mid prompt-build — an absent body has
    # no backtick runs, so the minimum fence applies.
    runs = re.findall(r"`+", text or "")
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)
