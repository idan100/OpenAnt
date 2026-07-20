"""Regression test for the "Retry 19/10" logging bug in core/analyzer.py.

``retryable_indices`` holds ORIGINAL unit positions in the full scan
(e.g. unit #18 out of 30 total units, of which only 10 need retrying).
The retry loop used to print the unit's own index as if it were a
sequential attempt counter against the retryable-count denominator —
"Retry 19/10" for unit index 18 (1-indexed) out of 10 retryable units.
There was never an actual retry-LIMIT being exceeded; this was purely
a mislabeled progress line. ``_retry_attempts`` fixes it by pairing
each original index with a genuine 1..N attempt sequence.
"""

from __future__ import annotations

from core.analyzer import _reclassify_error_breakdown, _retry_attempts


def test_attempt_numbers_are_sequential_regardless_of_original_indices():
    # The exact shape from the report: sparse original indices scattered
    # through a larger scan, several past the retryable count itself.
    retryable_indices = [3, 8, 18, 26]
    pairs = _retry_attempts(retryable_indices)
    attempts = [attempt for attempt, _ in pairs]
    assert attempts == [1, 2, 3, 4], "attempt numbers must never exceed len(retryable_indices)"


def test_original_indices_are_preserved_for_actual_unit_lookup():
    retryable_indices = [3, 8, 18, 26]
    pairs = _retry_attempts(retryable_indices)
    original_indices = [i for _, i in pairs]
    assert original_indices == retryable_indices


def test_single_retryable_unit():
    assert _retry_attempts([18]) == [(1, 18)]


def test_no_retryable_units():
    assert _retry_attempts([]) == []


def test_max_attempt_never_exceeds_total_count():
    """The exact reported symptom: 'Retry 19/10' — an attempt number
    exceeding the denominator. Must never happen for any input shape."""
    for retryable_indices in ([0], [5, 19], [3, 8, 18, 26], list(range(100, 110))):
        pairs = _retry_attempts(retryable_indices)
        total = len(retryable_indices)
        assert all(attempt <= total for attempt, _ in pairs), (
            f"attempt exceeded total {total} for indices {retryable_indices}: {pairs}"
        )


# ---------------------------------------------------------------------------
# _reclassify_error_breakdown — keeps the per-category error breakdown
# consistent with the total error count across a retry pass, including
# when a retry's error RECLASSIFIES (different pool candidate, different
# failure type) rather than just succeeding or failing identically.
# ---------------------------------------------------------------------------


def test_retry_success_removes_from_old_bucket():
    breakdown = {"connection": 1}
    _reclassify_error_breakdown(breakdown, "connection", None)
    assert breakdown == {}


def test_retry_still_fails_same_bucket_stays_put():
    breakdown = {"rate_limit": 1}
    _reclassify_error_breakdown(breakdown, "rate_limit", "rate_limit")
    assert breakdown == {"rate_limit": 1}


def test_retry_still_fails_but_reclassifies_to_a_different_bucket():
    # The scenario the fix exists for: attempt 1 was a connection error,
    # the retry landed on a different pool candidate and came back as a
    # malformed response instead.
    breakdown = {"connection": 1}
    _reclassify_error_breakdown(breakdown, "connection", "malformed_response")
    assert breakdown == {"malformed_response": 1}


def test_bucket_removed_entirely_once_its_count_hits_zero():
    breakdown = {"auth": 1, "connection": 3}
    _reclassify_error_breakdown(breakdown, "auth", None)
    assert "auth" not in breakdown
    assert breakdown == {"connection": 3}


def test_does_not_go_negative_if_old_bucket_already_empty():
    # Defensive: an old_bucket the breakdown doesn't actually have
    # (shouldn't happen in practice, but must not corrupt state).
    breakdown = {"connection": 1}
    _reclassify_error_breakdown(breakdown, "auth", "rate_limit")
    assert breakdown == {"connection": 1, "rate_limit": 1}
    assert "auth" not in breakdown


def test_multiple_retries_keep_breakdown_consistent_with_total():
    # Simulates a retry pass across 3 originally-errored units with
    # mixed outcomes: one succeeds, one reclassifies, one fails identically.
    breakdown = {"connection": 2, "rate_limit": 1}
    total_errors = 3

    _reclassify_error_breakdown(breakdown, "connection", None)  # succeeded
    total_errors -= 1
    _reclassify_error_breakdown(breakdown, "connection", "malformed_response")  # reclassified
    _reclassify_error_breakdown(breakdown, "rate_limit", "rate_limit")  # still failing

    assert sum(breakdown.values()) == total_errors
    assert breakdown == {"malformed_response": 1, "rate_limit": 1}
