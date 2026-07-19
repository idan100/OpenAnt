"""Active round-robin load balancing across multiple provider candidates.

Complements :class:`~.failover.FailoverAdapter`. Failover is
sequential and reactive: use the primary until it's exhausted, THEN
permanently switch to one backup. A pool is concurrent and proactive:
spread every call across N candidates that are ALL healthy right now,
so a phase draws on several SEPARATE quota pools simultaneously
instead of treating the others as pure backup — the point when the
goal is maximum aggregate throughput across e.g. a Claude subscription
plus several free-tier API keys, not just resilience against one of
them running out.

Lives entirely inside adapter construction
(``registry.build_phase_registry``) — no pipeline call site needs to
know pools exist; they keep calling
``binding.adapter.complete(model=binding.model, ...)`` exactly as
before. A pool ignores the caller-supplied ``model`` just like
:class:`~.failover.FailoverAdapter` does, since each candidate carries
its own model.

Composes with failover: a phase's PRIMARY side can itself be a pool
(e.g. Claude subscription + Gemini + Groq + OpenRouter, round-robined),
and the whole pool can still fail over to a configured fallback config
if literally every candidate in it is exhausted at once — see
``registry.build_phase_registry`` for how the two wrap each other.

Sticky within one multi-turn conversation. Tool-calling phases
(``enhance``, ``verify``) call ``complete()`` once per TURN of the same
back-and-forth, replaying the full ``messages`` history (including
earlier assistant turns' tool_use blocks) on every call. Rotating the
candidate on every one of those calls means turn 2 can land on a
DIFFERENT provider than turn 1 — which then has to replay a tool call
it never made itself. For Gemini specifically this is a hard 400
("Function call is missing a thought_signature"): the signature is
provider-specific and only exists for the provider that actually made
the call. Other providers likely have their own, less loudly-visible
failure modes replaying another provider's tool_use history.

Fixed by keying the rotation on conversation identity rather than call
count: a call whose ``messages`` already contains an assistant turn is
a CONTINUATION (reuse whatever candidate handled turn 1), not a fresh
pick. Each ThreadPoolExecutor worker processes one unit's conversation
to completion, synchronously, before picking up the next — so a
``threading.local`` pin is exactly conversation-scoped: the next unit's
first call always starts with a single fresh user message (no
assistant turn yet), which naturally reads as "new conversation" and
advances the rotation again. Single-shot phases (analyze, app_context,
report, ...) never carry an assistant turn either, so they round-robin
on every call exactly as before — this only changes behavior for
actual multi-turn tool loops.
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

from ..adapter import LLMAdapter, LLMError, LLMRateLimitError
from ...rate_limiter import get_rate_limiter, get_rpm_pacer


class PoolAdapter:
    """:class:`LLMAdapter` that round-robins across N (adapter, model,
    provider_name) candidates for one phase.

    ``supports_tools`` requires every candidate to support tools —
    checked once at registry-build time (mirrors
    :class:`~.failover.FailoverAdapter`), so a tool-calling phase never
    silently loses tool support mid-scan because a rotation happened
    to land on a non-tool-calling candidate.
    """

    def __init__(self, candidates: list[tuple[LLMAdapter, str, str]], phase: str) -> None:
        if not candidates:
            raise ValueError("PoolAdapter requires at least one candidate")
        self._candidates = candidates  # [(adapter, model, provider_name), ...]
        self._phase = phase
        self._lock = threading.Lock()
        self._next_index = 0
        # Per-thread pin so a multi-turn tool-calling conversation stays
        # on the SAME candidate across turns — see the module docstring's
        # "Sticky within one multi-turn conversation" section.
        self._sticky = threading.local()

    # ------------------------------------------------------------------
    # LLMAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        # Not a real provider identity — a human-readable label for
        # logging only. Rate-limit/RPM-pacer keying and probe-cache
        # lookups always use each CANDIDATE's own real (adapter.name,
        # model), never this joined string.
        return "+".join(dict.fromkeys(c[2] for c in self._candidates))

    @property
    def supports_tools(self) -> bool:
        return all(adapter.supports_tools for adapter, _, _ in self._candidates)

    @property
    def pricing(self) -> dict:
        # Best-effort union so a pricing lookup succeeds for whichever
        # candidate actually served a given call. Real per-call cost
        # attribution still keys off PhaseBinding.model, which is a
        # documented limitation shared with FailoverAdapter (see that
        # module's docstring) — this just avoids an unnecessary
        # "unknown model" warning on top of it.
        merged: dict = {}
        for adapter, _, _ in self._candidates:
            merged.update(getattr(adapter, "pricing", None) or {})
        return merged

    def _rotate_start(self) -> int:
        with self._lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % len(self._candidates)
            return idx

    def complete(self, *, model: str, **kwargs):  # noqa: ARG002 - see class docstring
        messages = kwargs.get("messages") or []
        is_continuation = any(getattr(m, "role", None) == "assistant" for m in messages)
        pinned_index = getattr(self._sticky, "index", None)

        if is_continuation and pinned_index is not None:
            # Mid-conversation: stay on the pinned candidate even if
            # it's currently rate-limited (it'll just block on its own
            # pacer/backoff) — falling through to a different provider
            # here would replay this provider's tool-call history
            # against one that never made those calls. See the module
            # docstring's "Sticky within one multi-turn conversation".
            adapter, candidate_model, provider_name = self._candidates[pinned_index]
            sys.stderr.write(f"[pool:{self._phase}] -> {provider_name}/{candidate_model} (sticky)\n")
            return adapter.complete(model=candidate_model, **kwargs)

        start = self._rotate_start()
        n = len(self._candidates)

        ready: list[tuple[int, LLMAdapter, str, str]] = []
        busy: list[tuple[int, LLMAdapter, str, str]] = []
        for offset in range(n):
            idx = (start + offset) % n
            adapter, candidate_model, provider_name = self._candidates[idx]
            if get_rate_limiter(provider_name).is_in_backoff():
                busy.append((idx, adapter, candidate_model, provider_name))
                continue
            pacer = get_rpm_pacer(provider_name, candidate_model)
            if pacer is not None and not pacer.has_immediate_slot():
                busy.append((idx, adapter, candidate_model, provider_name))
                continue
            ready.append((idx, adapter, candidate_model, provider_name))

        # Try candidates with no known reason to block first, in
        # rotation order; only fall through to a busy one (which will
        # then block on ITS OWN pacer/backoff) if every candidate is
        # currently busy — "just wait" is still better than erroring.
        last_exc: Optional[LLMRateLimitError] = None
        for idx, adapter, candidate_model, provider_name in ready + busy:
            sys.stderr.write(f"[pool:{self._phase}] -> {provider_name}/{candidate_model}\n")
            try:
                result = adapter.complete(model=candidate_model, **kwargs)
                # Pin so any FOLLOW-UP turn of this same conversation
                # (this thread processes one unit's conversation to
                # completion before picking up the next) stays here.
                self._sticky.index = idx
                return result
            except LLMRateLimitError as exc:
                last_exc = exc
                sys.stderr.write(
                    f"[pool:{self._phase}]    {provider_name}/{candidate_model} "
                    f"rate-limited, trying next candidate\n"
                )
                continue
        assert last_exc is not None  # unreachable with >=1 candidate; satisfies type-checkers
        raise last_exc

    def validate(self, model: str) -> None:  # noqa: ARG002 - see class docstring
        """Probe every candidate. Lenient: a pool member that's
        transiently down (e.g. a congested free-tier upstream) doesn't
        block scan startup as long as at least one candidate works —
        the pool naturally routes around a bad member at call time.
        Raises only if EVERY candidate fails.
        """
        from ..probe_cache import mark_validated, was_recently_validated

        successes = 0
        last_exc: Optional[LLMError] = None
        for adapter, candidate_model, provider_name in self._candidates:
            if was_recently_validated(adapter.name, candidate_model):
                successes += 1
                continue
            try:
                adapter.validate(candidate_model)
                mark_validated(adapter.name, candidate_model)
                successes += 1
            except LLMError as exc:
                last_exc = exc
                sys.stderr.write(
                    f"warning: pool member {provider_name}/{candidate_model} "
                    f"for phase {self._phase!r} failed validation "
                    f"({type(exc).__name__}: {exc}); it will be skipped at "
                    f"call time until it works again.\n"
                )
        if successes == 0:
            raise last_exc
