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

When EVERY candidate is currently busy (rate-limited or out of RPM
slots), ``complete()`` does not commit to trying them in fixed
rotation order — that just means blocking on whichever one the
rotation happens to land on first, which from the outside looks
identical to "the same model, over and over, failing" even though
each individual wait is technically correct. Instead it checks who
will clear soonest (a non-blocking peek — see ``_candidate_wait_seconds``)
and sleeps exactly that long, then re-evaluates from scratch, so a
DIFFERENT, healthier candidate gets a chance if one exists. Bounded to
candidates clearing within ``_MAX_CENTRAL_WAIT_SECONDS`` and
``_MAX_CENTRAL_WAIT_ROUNDS`` rounds — beyond that it falls through to
the old best-effort dispatch rather than stalling indefinitely.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from typing import Optional

from ..adapter import LLMAdapter, LLMError, LLMRateLimitError
from ...rate_limiter import get_rate_limiter, get_rpm_pacer

# "the next minute" — see complete()'s "every candidate busy" handling.
_MAX_CENTRAL_WAIT_SECONDS = 60.0
# Bounded so a pool where backoffs keep getting extended (siblings
# hitting fresh 429s while we wait) still eventually falls through to
# best-effort dispatch instead of stalling here indefinitely.
_MAX_CENTRAL_WAIT_ROUNDS = 5


def _candidate_wait_seconds(provider_name: str, model: str) -> float:
    """Non-blocking: seconds until this (provider, model) candidate
    could plausibly serve a request right now — the longer of its
    global-backoff remaining and its RPM pacer's remaining slot wait,
    0 if neither currently blocks it.
    """
    backoff_wait = get_rate_limiter(provider_name).time_until_ready()
    pacer = get_rpm_pacer(provider_name, model)
    pacer_wait = pacer.time_until_slot() if pacer is not None else 0.0
    return max(backoff_wait, pacer_wait)
from ...token_estimate import CHARS_PER_TOKEN, max_request_tokens_for


def _estimate_request_tokens(system, messages) -> int:
    """Rough size of one ``complete()`` call — duck-typed against the
    unified ``Message``/content-block shapes without importing them, so
    this stays decoupled from the concrete adapter contract the same
    way the rest of this generic ``**kwargs``-passthrough class does.
    """
    total_chars = len(system) if isinstance(system, str) else 0
    for message in messages or []:
        for block in getattr(message, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                total_chars += len(text)
            content = getattr(block, "content", None)
            if isinstance(content, str):
                total_chars += len(content)
    return total_chars // CHARS_PER_TOKEN


class _PoolPricing(dict):
    """Lazily queries each candidate's OWN ``pricing.get(model)`` on
    lookup rather than eagerly merging into a plain dict — see
    ``PoolAdapter.pricing``'s docstring for why eager merging silently
    drops a candidate like claude_sub's ``_ZeroCostPricing``. Empty by
    construction (``dict.__init__`` never called with real items); all
    real behavior lives in ``get()``.
    """

    def __init__(self, candidates: list[tuple[LLMAdapter, str, str]]):
        super().__init__()
        self._candidates = candidates

    def get(self, key, default=None):  # noqa: ANN001 - dict-compatible signature
        for adapter, _, _ in self._candidates:
            price = getattr(adapter, "pricing", None)
            if price is None:
                continue
            found = price.get(key)
            if found is not None:
                return found
        return default


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
    def pricing(self) -> _PoolPricing:
        # Best-effort union so a pricing lookup succeeds for whichever
        # candidate actually served a given call. Real per-call cost
        # attribution still keys off PhaseBinding.model, which is a
        # documented limitation shared with FailoverAdapter (see that
        # module's docstring) — this just avoids an unnecessary
        # "unknown model" warning on top of it.
        #
        # Deliberately NOT an eager dict.update() merge: claude_sub's
        # ``_ZeroCostPricing`` reports $0 via a custom ``.get()``
        # override with no real stored keys (by design — subscription
        # billing has no marginal cost), so merging it via
        # ``dict.update()`` (which only copies STORED items) silently
        # drops it — the moment claude_sub is a pool member rather than
        # a standalone adapter, ``lookup_pricing()`` stops seeing it and
        # fires a spurious "no pricing for model 'opus'" warning. Lazily
        # querying each candidate's OWN ``.get()`` instead respects
        # whatever pricing behavior that candidate actually implements.
        return _PoolPricing(self._candidates)

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

        n = len(self._candidates)
        estimated_tokens = _estimate_request_tokens(kwargs.get("system"), messages)

        for _round in range(_MAX_CENTRAL_WAIT_ROUNDS + 1):
            start = self._rotate_start()

            ready: list[tuple[int, LLMAdapter, str, str]] = []
            busy: list[tuple[int, LLMAdapter, str, str]] = []
            # Candidates with a KNOWN per-request token ceiling this
            # call would exceed (see utilities/token_estimate.py) —
            # tried only as an absolute last resort. Unlike "busy",
            # waiting never helps here: the request is simply too big
            # for that provider's tier, so it's worse odds than a
            # rate-limited candidate that will clear on its own.
            oversized: list[tuple[int, LLMAdapter, str, str]] = []
            for offset in range(n):
                idx = (start + offset) % n
                adapter, candidate_model, provider_name = self._candidates[idx]
                ceiling = max_request_tokens_for(provider_name, candidate_model)
                if ceiling is not None and estimated_tokens > ceiling:
                    oversized.append((idx, adapter, candidate_model, provider_name))
                    continue
                if get_rate_limiter(provider_name).is_in_backoff():
                    busy.append((idx, adapter, candidate_model, provider_name))
                    continue
                pacer = get_rpm_pacer(provider_name, candidate_model)
                if pacer is not None and not pacer.has_immediate_slot():
                    busy.append((idx, adapter, candidate_model, provider_name))
                    continue
                ready.append((idx, adapter, candidate_model, provider_name))

            if ready or not busy or _round == _MAX_CENTRAL_WAIT_ROUNDS:
                break

            # EVERY candidate is busy. Committing to rotation order here
            # means blocking on whichever one the rotation happens to
            # land on first — often the SAME struggling candidate every
            # time, which is indistinguishable from "the same model
            # over and over, failing" from the outside, even though
            # each individual wait is technically correct. Instead:
            # check who will actually clear soonest and sleep exactly
            # that long, then re-evaluate from scratch — a different,
            # healthier candidate may now be ready, not just the one we
            # waited for.
            soonest = min(
                _candidate_wait_seconds(provider_name, candidate_model)
                for _, _, candidate_model, provider_name in busy
            )
            if soonest > _MAX_CENTRAL_WAIT_SECONDS:
                break  # nothing clearing soon enough — fall through to best-effort dispatch below
            sys.stderr.write(
                f"[pool:{self._phase}] all {len(busy)} candidate(s) busy — "
                f"waiting {soonest:.0f}s for the soonest to free up "
                f"(not blindly retrying the same one)\n"
            )
            time.sleep(soonest + random.uniform(0.05, 1.0))

        # Try candidates with no known reason to block first, in
        # rotation order; fall through to a busy one (which will then
        # block on ITS OWN pacer/backoff) before an oversized one —
        # "just wait" beats "definitely too big" — and only reach
        # oversized if literally nothing else is available.
        last_exc: Optional[LLMRateLimitError] = None
        for idx, adapter, candidate_model, provider_name in ready + busy + oversized:
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
