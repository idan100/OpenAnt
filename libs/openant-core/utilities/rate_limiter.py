"""
Process-level rate limiter with coordinated backoff, keyed per provider.

When any worker hits a 429 rate limit error, ALL workers ON THE SAME
PROVIDER pause for a configurable backoff period (default 30s). This
prevents thundering herd and ensures the rate limit window has time to
reset. Providers are isolated from each other: a Gemini 429 does not
pause Anthropic/OpenAI workers, since they draw on entirely separate
quota — see ``get_rate_limiter(provider=...)`` below. Workers sharing
one provider still coordinate perfectly (the original PR #69 fix this
module exists for).

Usage:
    from utilities.rate_limiter import get_rate_limiter, configure_rate_limiter

    # At startup (once) - sets the default backoff for every provider
    configure_rate_limiter(backoff_seconds=30)

    # Before every API call - one limiter instance per provider name
    rate_limiter = get_rate_limiter("anthropic")
    rate_limiter.wait_if_needed()

    # When catching RateLimitError
    except anthropic.RateLimitError as e:
        retry_after = float(e.response.headers.get("retry-after", 0))
        rate_limiter.report_rate_limit(retry_after)
        raise
"""

import collections
import random
import sys
import threading
import time


class GlobalRateLimiter:
    """
    Rate limiter with coordinated backoff across all threads sharing
    this instance. Callers get one instance per provider name via the
    module-level registry below, so coordination stays scoped to
    workers hitting the SAME provider's quota.
    """

    def __init__(self, backoff_seconds: float = 30.0):
        self._lock = threading.Lock()
        self._backoff_until = 0.0
        self._backoff_seconds = backoff_seconds
        self._total_waits = 0
        self._total_wait_time = 0.0

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_seconds

    @backoff_seconds.setter
    def backoff_seconds(self, value: float):
        self._backoff_seconds = value

    def wait_if_needed(self) -> float:
        """
        Block if currently in a backoff period.

        Call this before every API request. Returns the time waited (0 if none).
        """
        total_wait = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._backoff_until:
                    break

                wait_time = self._backoff_until - now
                # Add jitter (0-2s) to prevent thundering herd when backoff expires
                jitter = random.uniform(0, 2.0)
                this_wait = wait_time + jitter

            # Sleep outside the lock so other threads can also read backoff_until.
            # Re-check after sleeping: another worker may have EXTENDED _backoff_until
            # (a fresh 429 via report_rate_limit) while we slept. Without the re-check we
            # would wake into the still-active backoff window and re-trigger the storm.
            time.sleep(this_wait)
            total_wait += this_wait

        if total_wait > 0.0:
            with self._lock:
                self._total_waits += 1
                self._total_wait_time += total_wait

        return total_wait

    def report_rate_limit(self, retry_after: float | None = None):
        """
        Report a rate limit error and trigger global backoff.

        Call this when any worker receives a 429 error. All workers will
        pause until the backoff period expires.

        Args:
            retry_after: The retry-after header value from the API response.
                If provided, uses max(retry_after, backoff_seconds).
        """
        with self._lock:
            # Use the larger of retry_after and our configured backoff
            backoff = max(retry_after or 0.0, self._backoff_seconds)
            new_backoff_until = time.monotonic() + backoff

            # Only extend if this is later than current backoff
            if new_backoff_until > self._backoff_until:
                self._backoff_until = new_backoff_until
                print(
                    f"[RateLimiter] Global backoff triggered: {backoff:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

    def is_in_backoff(self) -> bool:
        """Check if currently in a backoff period (for diagnostics)."""
        with self._lock:
            return time.monotonic() < self._backoff_until

    def time_until_ready(self) -> float:
        """Seconds until backoff expires (0 if not in backoff)."""
        with self._lock:
            remaining = self._backoff_until - time.monotonic()
            return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get statistics about rate limiting (for diagnostics)."""
        with self._lock:
            return {
                "total_waits": self._total_waits,
                "total_wait_time": round(self._total_wait_time, 2),
                "backoff_seconds": self._backoff_seconds,
                "currently_in_backoff": time.monotonic() < self._backoff_until,
            }

    def reset(self):
        """Reset backoff state. For testing."""
        with self._lock:
            self._backoff_until = 0.0
            self._total_waits = 0
            self._total_wait_time = 0.0


# Module-level registry: one GlobalRateLimiter instance per provider
# name, so a rate-limit pause on one provider doesn't stall workers on
# an unrelated provider's quota.
_DEFAULT_PROVIDER = "default"
_rate_limiters: dict[str, GlobalRateLimiter] = {}
_default_backoff_seconds = 30.0
_config_lock = threading.Lock()


def configure_rate_limiter(
    backoff_seconds: float = 30.0, provider: str = _DEFAULT_PROVIDER
) -> GlobalRateLimiter:
    """
    Configure a provider's rate limiter. Call once per provider at startup
    (or once with no ``provider`` to set the process-wide default new
    provider limiters pick up when first created).

    Args:
        backoff_seconds: How long to pause this provider's workers on
            rate limit (default: 30s).
        provider: Provider name (e.g. "anthropic", "google"). Omit to
            configure the shared default — this ALSO updates every
            already-created provider limiter's backoff duration, matching
            the pre-multi-provider behavior of one shared setting.

    Returns:
        The configured GlobalRateLimiter for ``provider``.
    """
    global _default_backoff_seconds
    with _config_lock:
        if provider == _DEFAULT_PROVIDER:
            _default_backoff_seconds = backoff_seconds
            for limiter in _rate_limiters.values():
                limiter.backoff_seconds = backoff_seconds
        limiter = _rate_limiters.get(provider)
        if limiter is None:
            limiter = GlobalRateLimiter(backoff_seconds)
            _rate_limiters[provider] = limiter
        else:
            limiter.backoff_seconds = backoff_seconds
        return limiter


def get_rate_limiter(provider: str = _DEFAULT_PROVIDER) -> GlobalRateLimiter:
    """
    Get the rate limiter for ``provider``.

    If not yet configured, creates one using the process-wide default
    backoff (30s unless changed via ``configure_rate_limiter()``).
    """
    with _config_lock:
        limiter = _rate_limiters.get(provider)
        if limiter is None:
            limiter = GlobalRateLimiter(_default_backoff_seconds)
            _rate_limiters[provider] = limiter
        return limiter


def reset_rate_limiter():
    """Reset every provider's rate limiter. For testing."""
    with _config_lock:
        for limiter in _rate_limiters.values():
            limiter.reset()


class RpmPacer:
    """Proactive sliding-window request pacer for a fixed RPM ceiling.

    Complements ``GlobalRateLimiter``: that class reacts to a 429
    AFTER it happens (coordinated backoff). This paces requests BEFORE
    they're sent so a known-tiny ceiling (e.g. a Gemini free-tier
    model's 5-15 RPM) is rarely exceeded in the first place — several
    parallel workers hammering a 5 RPM model otherwise 429 almost
    immediately and repeatedly re-trigger the reactive backoff, which
    is slower overall than just spacing requests out to begin with.

    Enforces "no more than N requests in any trailing 60-second
    window" by tracking request timestamps in a deque: once N are
    in flight within the window, a new caller blocks until the
    oldest one ages out.
    """

    def __init__(self, rpm_limit: float):
        self._limit = max(1, round(rpm_limit))
        self._lock = threading.Lock()
        self._timestamps: collections.deque[float] = collections.deque()

    def wait_for_slot(self) -> float:
        """Block until a request is safe to send under the RPM ceiling.

        Returns the time waited (0 if none). Records the slot as used
        immediately upon returning — callers should call this
        immediately before issuing the request, not speculatively.
        """
        total_wait = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return total_wait
                # Oldest slot ages out of the 60s window at this time.
                wait_time = 60.0 - (now - self._timestamps[0])
            # Sleep outside the lock so other threads can also check in.
            # Small jitter avoids every blocked worker waking at the
            # exact same instant and re-contending for the same slot.
            this_wait = max(0.05, wait_time) + random.uniform(0, 0.5)
            time.sleep(this_wait)
            total_wait += this_wait

    def has_immediate_slot(self) -> bool:
        """Non-blocking peek: would ``wait_for_slot()`` return immediately?

        Does NOT claim a slot — purely informational, for a round-robin
        pool deciding which candidate to try first (see
        ``providers/pool.py``). A slot can still be taken by another
        thread between this call and an actual ``wait_for_slot()`` —
        harmless race: worst case the caller picks a candidate that
        then blocks briefly, no worse than not checking at all.
        """
        with self._lock:
            now = time.monotonic()
            active = sum(1 for t in self._timestamps if now - t < 60.0)
            return active < self._limit

    def time_until_slot(self) -> float:
        """Non-blocking: seconds until ``wait_for_slot()`` would return
        (0 if it would return immediately right now).

        Companion to :meth:`GlobalRateLimiter.time_until_ready` — lets
        a caller decide WHOM to wait for before committing to a sleep,
        rather than blocking blind inside ``wait_for_slot()`` on
        whichever candidate it happened to try first (see
        ``providers/pool.py``'s "all candidates busy" handling).
        """
        with self._lock:
            now = time.monotonic()
            active = [t for t in self._timestamps if now - t < 60.0]
            if len(active) < self._limit:
                return 0.0
            return max(0.0, 60.0 - (now - active[0]))

    def reset(self) -> None:
        """Clear tracked request history. For testing."""
        with self._lock:
            self._timestamps.clear()


# Module-level registry: one RpmPacer per (provider, model) — RPM
# ceilings are a property of a specific model, not a whole provider
# (a single "google" provider can span models with very different
# limits — see the ``gemini`` example config). None registered for a
# given (provider, model) means no proactive pacing — current,
# reactive-only behavior, unchanged unless a caller opts in.
_rpm_pacers: dict[tuple[str, str], RpmPacer] = {}
_rpm_config_lock = threading.Lock()


def configure_rpm_limit(provider: str, model: str, rpm_limit: float | None) -> None:
    """Register (or clear) a proactive RPM ceiling for (provider, model).

    Called once per unique (provider, model) at registry-build time —
    see ``utilities/llm/registry.py``. ``rpm_limit=None`` removes any
    existing pacer for this key (no proactive pacing).
    """
    key = (provider, model)
    with _rpm_config_lock:
        if rpm_limit is None:
            _rpm_pacers.pop(key, None)
        else:
            _rpm_pacers[key] = RpmPacer(rpm_limit)


def get_rpm_pacer(provider: str, model: str) -> RpmPacer | None:
    """Return the configured pacer for (provider, model), or None."""
    with _rpm_config_lock:
        return _rpm_pacers.get((provider, model))


def reset_rpm_pacers() -> None:
    """Clear every configured RPM pacer. For testing."""
    with _rpm_config_lock:
        _rpm_pacers.clear()


def is_rate_limit_error(error_info: dict | str | None) -> bool:
    """
    Check if an error dict/string represents a rate limit error.

    Args:
        error_info: The error field from agent_context or similar.

    Returns:
        True if this is a rate limit error that should be retried.
    """
    if not error_info:
        return False
    if isinstance(error_info, dict):
        return error_info.get("type") == "rate_limit"
    return "rate_limit" in str(error_info).lower()


def is_retryable_error(error_info: dict | str | None) -> bool:
    """
    Check if an error is retryable (transient network/server issues).

    Retryable errors include:
    - rate_limit: API rate limiting (429)
    - connection: Network connectivity issues
    - timeout: Request timeout
    - api_status with 500+: Server errors (not client errors like 400)

    Args:
        error_info: The error field from agent_context or similar.

    Returns:
        True if this error should be retried.
    """
    if not error_info:
        return False
    
    if isinstance(error_info, dict):
        error_type = error_info.get("type", "")
        
        # Always retry these transient error types
        if error_type in ("rate_limit", "connection", "timeout"):
            return True
        
        # Retry server errors (5xx), but not client errors (4xx)
        if error_type == "api_status":
            status_code = error_info.get("status_code", 0)
            return status_code >= 500
        
        return False
    
    # String-based error checking.
    # NOTE: the analyzer detection path (core/analyzer.py) stores the raw
    # str(e) of an exception rather than a structured dict, so an Anthropic
    # HTTP 529 surfaces here as e.g.
    #   "Error code: 529 - {...'type':'overloaded_error'...}"
    # 529 ("overloaded") is the most common transient Anthropic failure under
    # load and is a 5xx, so it must be retried. The structured dict branch
    # above already retries it via status_code >= 500; mirror that here so the
    # string path is not silently non-retryable.
    error_str = str(error_info).lower()
    return any(term in error_str for term in (
        "rate_limit", "connection", "timeout",
        "500", "502", "503", "504", "529", "overloaded",
        # Provider-reported failure states (finish_reason='error' /
        # 'tool_use_failed') and detected pseudo-tool-call syntax
        # leakage — see utilities/llm/providers/openai.py. A fresh
        # retry gets a NEW conversation, which (via PoolAdapter's
        # per-conversation stickiness) can land on a different, less
        # broken candidate — often enough to be worth retrying.
        "finish_reason='error'", "finish_reason='tool_use_failed'",
        "malformed_tool_syntax",
        # Gemini's tool-calling failure states (see google.py) — matched
        # without a "finish_reason=" prefix since the SDK sometimes
        # stringifies the enum as "FinishReason.X" rather than bare "X".
        "malformed_function_call", "unexpected_tool_call",
    ))
