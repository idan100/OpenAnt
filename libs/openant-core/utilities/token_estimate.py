"""Rough token-size estimation and known per-request provider ceilings.

No tiktoken (or equivalent tokenizer) dependency exists in this repo, and
adding one just for a size guard is unwarranted given how approximate this
already is — providers use different tokenizers, so ANY estimate is a
heuristic, not a precise count. Uses the same ~4 chars/token assumption
``prompts/_fence.py:cap_code`` already documents, so the two stay
consistent about what "big" means.

``KNOWN_MAX_REQUEST_TOKENS`` exists because a "413 Request body too large"
observed against GitHub Models' ``openai/gpt-4.1-mini`` turned out to be a
per-request cap tied to that PROVIDER'S FREE-TIER ACCESS LEVEL (GitHub's
"Low" tier: 8k in / 4k out), not the model's real ~1M context window —
nothing in ``ProviderConfig``/``PhaseRef`` distinguishes "true model
context window" from "this provider's tier-imposed request cap" today, so
this table is a narrow, explicit list of observed exceptions rather than a
general per-model context-window database (which would need to be kept in
sync with every provider's own, frequently-changing tier limits — a much
bigger undertaking than fixing the one confirmed failure).
"""

from __future__ import annotations

# ~4 characters per token — matches prompts/_fence.py:cap_code's own
# documented assumption. Exported (not module-private) so callers that
# already have a char COUNT rather than the text itself (e.g. summing
# lengths across many blocks) can divide directly without building a
# throwaway string just to call estimate_tokens().
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count for ``text``. Approximate by design — see module docstring."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


# (provider_name, model) -> max total request tokens (input side).
# Extend this as other tight provider-tier limits are confirmed.
KNOWN_MAX_REQUEST_TOKENS: dict[tuple[str, str], int] = {
    ("github", "openai/gpt-4.1-mini"): 8_000,
    ("github", "openai/gpt-4o-mini"): 8_000,
}


def max_request_tokens_for(provider_name: str, model: str) -> "int | None":
    """Known per-request token ceiling for ``(provider_name, model)``, or
    ``None`` when nothing narrower than the model's real context window
    is known to apply."""
    return KNOWN_MAX_REQUEST_TOKENS.get((provider_name, model))
