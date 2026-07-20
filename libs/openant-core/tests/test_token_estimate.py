"""Tests for the rough token-size estimator and known per-request
provider ceilings used by PoolAdapter's context-window routing."""

from __future__ import annotations

from utilities.token_estimate import estimate_tokens, max_request_tokens_for


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_roughly_chars_over_four():
    assert estimate_tokens("x" * 400) == 100


def test_max_request_tokens_for_known_entry():
    assert max_request_tokens_for("github", "openai/gpt-4.1-mini") == 8_000


def test_max_request_tokens_for_unknown_entry_is_none():
    assert max_request_tokens_for("groq", "llama-3.3-70b-versatile") is None
    assert max_request_tokens_for("nonexistent", "nonexistent") is None
