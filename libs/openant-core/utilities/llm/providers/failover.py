"""Transparent provider failover for a hard usage-cap mid-scan.

Wraps a phase's primary adapter so a hard-exhaustion signal (the
subscription's usage window is spent, a metered account's budget is
exhausted — as opposed to an ordinary transient 429) permanently
switches that call, and every subsequent one sharing the same primary
provider, to a configured fallback provider for the rest of the run.

This lives entirely inside adapter construction
(``registry.build_phase_registry``) — no pipeline call site
(``finding_verifier.py``, ``agentic_enhancer/agent.py``, etc.) needs to
know failover exists. They keep calling
``binding.adapter.complete(model=binding.model, ...)`` exactly as
before; ``FailoverAdapter`` intercepts, and once failed over, ignores
the caller-supplied ``model`` in favor of the fallback's own configured
model (the same way ``PhaseBinding.model`` already forces call sites
onto the configured model rather than something dynamic).

Known, accepted trade-off: ``PhaseBinding.model`` / ``.provider_name``
stay fixed at the PRIMARY's values even after failover (they're plain
dataclass fields, read by ~15 call sites across the pipeline for
logging and cost-tracking). Token counts stay accurate post-failover
(``CompletionResult`` always reports what the adapter that actually
served the call used); the $-cost estimate for calls made during an
active failover window may use the primary's price table instead of
the fallback's. ``ponytail:`` accepted ceiling — ``PhaseBinding.model``
becoming a live property would need every one of those call sites
touched for a rare event. Upgrade path if this starts to matter:
give ``FailoverAdapter`` a ``current_model`` property and thread it
through ``lookup_pricing``/``TokenTracker.record_call`` instead of
``binding.model``.
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

from ..adapter import LLMAdapter, LLMRateLimitError

# A rate-limit error with no retry hint, or one far longer than any
# sane short-backoff-and-retry window, reads as "the usage window
# itself is spent" rather than "wait a few seconds and try again".
# claude_subscription's session-cap error is the clearest example: it
# raises LLMRateLimitError with retry_after=None (see that adapter's
# module docstring / _ERROR_MESSAGES["rate_limit"]) because the Agent
# SDK carries no retry hint for a real subscription-cap hit.
_EXHAUSTION_RETRY_AFTER_THRESHOLD_SECONDS = 300.0

# Require more than one exhaustion-looking signal before actually
# flipping over. Guards against a single edge case slipping past the
# retry_after filter above (e.g. a genuinely transient 529 "overloaded"
# that happens to arrive with no retry-after header) triggering an
# unwanted, permanent switch off a provider that was actually fine.
_SIGNALS_REQUIRED_TO_FAIL_OVER = 2


def is_exhaustion_signal(exc: LLMRateLimitError) -> bool:
    """Heuristic: does this rate-limit error look like a hard usage cap?"""
    retry_after = exc.retry_after
    return retry_after is None or retry_after > _EXHAUSTION_RETRY_AFTER_THRESHOLD_SECONDS


class ExhaustionState:
    """Shared per-primary-provider failover flag.

    All :class:`FailoverAdapter` instances wrapping the SAME primary
    provider (e.g. every phase routed through ``claude_sub``) share one
    of these, so exhaustion discovered via ANY phase's call immediately
    protects the others — they don't each have to independently burn a
    failing call against an already-exhausted provider before noticing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failed_over = False
        self._signal_count = 0

    def is_failed_over(self) -> bool:
        with self._lock:
            return self._failed_over

    def report_signal(self) -> bool:
        """Record one hard-exhaustion-looking error.

        Returns True exactly once — the call that actually flips
        ``is_failed_over()`` to True — so the caller logs the switch a
        single time. Returns False otherwise (already failed over, or
        not enough signals yet).
        """
        with self._lock:
            if self._failed_over:
                return False
            self._signal_count += 1
            if self._signal_count >= _SIGNALS_REQUIRED_TO_FAIL_OVER:
                self._failed_over = True
                return True
            return False


class FailoverAdapter:
    """:class:`LLMAdapter` wrapper that fails over to a fallback provider.

    Constructed once per phase (the fallback (provider, model) pair can
    differ per phase even when the fallback PROVIDER is shared — see
    the ``gemini`` config's per-phase model split), but shares an
    :class:`ExhaustionState` with every other phase using the same
    primary provider.
    """

    def __init__(
        self,
        *,
        primary: LLMAdapter,
        primary_model: str,
        fallback: LLMAdapter,
        fallback_model: str,
        phase: str,
        exhaustion_state: ExhaustionState,
    ) -> None:
        self._primary = primary
        self._primary_model = primary_model
        self._fallback = fallback
        self._fallback_model = fallback_model
        self._phase = phase
        self._state = exhaustion_state

    # ------------------------------------------------------------------
    # LLMAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._fallback.name if self._state.is_failed_over() else self._primary.name

    @property
    def supports_tools(self) -> bool:
        # Both sides must support tools for this to report True — a
        # tool-required phase (enhance/verify) must stay usable no
        # matter which side is currently active. Checked once at
        # registry-build time by _check_tool_support; every shipped
        # adapter supports tools today, so this is a no-op in practice,
        # but a future non-tool-calling adapter used as a fallback for
        # a tool phase will be caught here instead of failing mid-scan.
        return self._primary.supports_tools and self._fallback.supports_tools

    @property
    def pricing(self) -> dict:
        # Reflects whichever side is currently active. See the module
        # docstring's accepted trade-off re: PhaseBinding.model staying
        # fixed on the primary's model string.
        active, _ = self._active()
        return active.pricing

    def complete(self, *, model: str, **kwargs):  # noqa: ARG002 - model comes from binding, see class docstring
        if self._state.is_failed_over():
            self._log_dispatch(self._fallback, self._fallback_model)
            return self._fallback.complete(model=self._fallback_model, **kwargs)
        self._log_dispatch(self._primary, self._primary_model)
        try:
            return self._primary.complete(model=self._primary_model, **kwargs)
        except LLMRateLimitError as exc:
            if not is_exhaustion_signal(exc):
                raise
            if self._state.report_signal():
                sys.stderr.write(
                    f"[failover] phase {self._phase!r}: {self._primary.name} "
                    f"usage appears exhausted ({exc}); switching to "
                    f"{self._fallback.name}/{self._fallback_model} for the "
                    f"rest of this run\n"
                )
            if not self._state.is_failed_over():
                # First signal, threshold not yet reached — propagate so
                # the pipeline's normal retry/backoff loop handles this
                # one exactly as it would without failover configured.
                raise
            return self._fallback.complete(model=self._fallback_model, **kwargs)

    def _log_dispatch(self, target: LLMAdapter, target_model: str) -> None:
        # Skip when the target is itself a PoolAdapter — it already
        # prints its own per-call "[pool:phase] -> provider/model" line
        # (see providers/pool.py), so this would just be a redundant
        # duplicate for every pooled phase's normal case.
        from .pool import PoolAdapter

        if isinstance(target, PoolAdapter):
            return
        sys.stderr.write(f"[phase:{self._phase}] -> {target.name}/{target_model}\n")

    def validate(self, model: str) -> None:  # noqa: ARG002 - model comes from binding, see class docstring
        from ..probe_cache import mark_validated, was_recently_validated

        active, active_model = self._active()
        # ``active.name`` is a real leaf identity for a plain adapter,
        # or a synthesized "provider+provider" label when ``active`` is
        # itself a PoolAdapter — in the pool case this cache check is a
        # harmless near-always-miss no-op, because PoolAdapter.validate()
        # already does correct PER-MEMBER caching internally.
        if was_recently_validated(active.name, active_model):
            return
        active.validate(active_model)
        mark_validated(active.name, active_model)

    def _active(self) -> tuple[LLMAdapter, str]:
        if self._state.is_failed_over():
            return self._fallback, self._fallback_model
        return self._primary, self._primary_model
