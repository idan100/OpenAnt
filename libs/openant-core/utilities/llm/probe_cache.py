"""Short-lived on-disk cache of successfully-probed (adapter, model) pairs.

``PhaseRegistry.validate()`` runs at the start of every scan/step verb
(``probe_registry_or_raise``) and, since the failover feature, ALSO
probes the configured fallback — meaning a Gemini free-tier key now
eats a handful of 1-token validation requests on every single
invocation, even when the exact same (adapter, model) validated fine a
minute ago. On a request-constrained free tier that's real, avoidable
quota spend. This cache lets a probe within the last 30 minutes be
skipped.

Deliberately NOT in config.json (that's user-authored credentials/
routing, not runtime cache state) and NOT in-memory-only (each CLI
invocation is a fresh Python process — an in-memory cache would never
survive between scans, which is exactly when this matters). Lives
beside config.json instead, keyed on the ADAPTER's type name (e.g.
"google"), not the user's provider entry name — so renaming a
provider in config.json doesn't spuriously invalidate the cache, and a
real credential/model change is still caught on the adapter's own
terms (nothing here inspects the api_key).

Worst case on a stale/wrong cache entry: a scan skips its upfront
probe and the FIRST real ``complete()`` call surfaces the actual
auth/model error instead — not silent, just slightly delayed. That
trade-off is worth the request savings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_TTL_SECONDS = 1800.0  # 30 minutes


def _cache_path() -> Path:
    # Mirrors registry.default_config_path()'s directory without
    # importing from .registry — that module calls into this one from
    # inside PhaseRegistry.validate() (a deferred, in-function import),
    # so importing registry at THIS module's top level would be
    # circular.
    import os

    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) / "openant" if xdg else Path.home() / ".config" / "openant"
    return base / "probe_cache.json"


def _load() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt/unreadable cache isn't worth failing a scan over —
        # treat it as empty and let probing proceed normally.
        return {}


def _save(data: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # best-effort; a failed write just means the next probe re-runs


def _key(adapter_name: str, model: str) -> str:
    return f"{adapter_name}:{model}"


def was_recently_validated(adapter_name: str, model: str) -> bool:
    last = _load().get(_key(adapter_name, model))
    if last is None:
        return False
    return (time.time() - last) < _TTL_SECONDS


def mark_validated(adapter_name: str, model: str) -> None:
    data = _load()
    data[_key(adapter_name, model)] = time.time()
    _save(data)
