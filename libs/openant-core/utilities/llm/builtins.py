"""The frozen ``openant-default`` llm-config.

Properties (per plan §7):

* **Source-defined, not on disk.** ``openant-default`` is the
  baked-in baseline that always resolves, even on a fresh install
  with no config.json.
* **Immutable.** ``parse_config()`` rejects any user attempt to
  redefine it. Users customise by copying it under a different name
  (``openant llm-config copy openant-default my-config``).
* **References provider name "anthropic".** The provider entry IS
  user-editable; this lets ``openant set-api-key`` write the key to
  ``llm_providers["anthropic"].api_key`` and have ``openant-default``
  pick it up automatically.

If Anthropic deprecates a model ID listed here, this file is the
single place we update — every other module reads through the
registry.
"""

from __future__ import annotations

from .config import PHASES, LLMConfig, PhaseRef


# Provider name referenced by every phase. Synthesised from the
# legacy ``api_key`` field by the migrator, or set via
# ``openant set-api-key``.
_ANTHROPIC_PROVIDER = "anthropic"


# Every phase on Claude Sonnet 5 — the user-requested default for this
# version of OpenAnt. Previously a per-phase Opus/Sonnet split (Opus for
# analyze/verify/llm_reach/report, Sonnet 4 for the rest); Sonnet 5 closes
# most of that reasoning gap at a fraction of Opus's per-token cost, so a
# single model everywhere is both simpler and cheaper.
# When this file changes, the CHANGELOG must say so, because every
# existing user without a custom llm-config picks up the new IDs on
# the next ``openant scan``.
OPENANT_DEFAULT = LLMConfig(
    name="openant-default",
    phases={
        phase: PhaseRef(provider=_ANTHROPIC_PROVIDER, model="claude-sonnet-5")
        for phase in PHASES
    },
)


# Public, callable accessor so callers don't accidentally mutate the
# module-level dict. The dataclass is frozen so the dict-mutation
# foot-gun is mostly hypothetical, but this gives us a single hook
# if we ever want to load the default from disk for testing.
def get_builtin_default() -> LLMConfig:
    return OPENANT_DEFAULT
