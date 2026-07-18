"""Shared rate-limiter glue for provider adapters.

Every adapter cooperates with a per-provider :class:`GlobalRateLimiter`
(keyed by ``provider`` — pass the adapter's own ``self.name``) so a
429/529 on any worker thread backs the *other workers on that same
provider* off. Centralised here so a new adapter can't silently skip
the dance.

That omission is exactly the H1 defect from PR #69: only the Anthropic
adapter called the limiter, so with 8 workers on a shared OpenAI/Google
quota, one worker's 429 left the other seven stampeding. Keying by
provider preserves that fix while stopping a DIFFERENT defect: a config
can route different pipeline phases to different providers running
concurrently in the same process (see README's hand-authored config
example), and those providers draw on entirely separate quota — a
Gemini 429 backing off Anthropic workers (or vice versa) wastes wall
clock for no reason. Wiring goes through these two helpers in every
adapter's ``complete()``:

    wait_for_rate_limit(self.name, model)          # before issuing the request
    ...
    except <provider 429>:
        report_rate_limit(self.name, retry_after)  # on the rate-limit branch

``wait_for_rate_limit`` ALSO consults a proactive per-(provider,
model) :class:`RpmPacer`, when one is configured (via
``configure_rpm_limit`` — see ``utilities/llm/registry.py``). That's
what keeps a known-tiny RPM ceiling (a Gemini free-tier model, say)
from being hit at all in the common case, rather than only reacting
to the 429 after the fact. ``model`` is optional so any future direct
caller that genuinely has no model in scope degrades to
backoff-only behavior instead of erroring.
"""

from __future__ import annotations

from typing import Optional

from ...rate_limiter import get_rate_limiter, get_rpm_pacer


def wait_for_rate_limit(provider: str, model: Optional[str] = None) -> None:
    """Block if a sibling worker on ``provider`` recently hit a 429/529,
    THEN block again if ``model`` has a configured RPM ceiling with no
    free slot right now.

    Call once at the top of ``complete()``, before the network request.
    """
    get_rate_limiter(provider).wait_if_needed()
    if model is not None:
        pacer = get_rpm_pacer(provider, model)
        if pacer is not None:
            pacer.wait_for_slot()


def report_rate_limit(provider: str, retry_after: Optional[float]) -> None:
    """Trigger backoff for ``provider`` after a 429/529.

    Call from the adapter's rate-limit ``except`` branch. ``retry_after``
    is the provider's hint in seconds (``None`` when absent — the limiter
    falls back to its configured default backoff).
    """
    get_rate_limiter(provider).report_rate_limit(retry_after or 0.0)
