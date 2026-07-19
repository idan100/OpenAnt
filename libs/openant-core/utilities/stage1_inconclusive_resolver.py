"""Generic resolution pass for Stage 1 "inconclusive" verdicts.

Stage 1 sometimes correctly recognizes it CAN'T prove a tainted value's
origin from the code it was given (see the origin-tracing requirement
in ``prompts/vulnerability_analysis.py``) and returns "inconclusive"
instead of guessing. That's the right call given what it could see —
but it isn't the end of the story: this repo's own repository index
often knows exactly who calls the analyzed function, even when that
caller wasn't in Stage 1's own context window. This module looks that
up deterministically — the same caller-search pattern used for Stage
2's ``FindingVerifier._find_sibling_call_sites`` — and, when it finds
something, gives Stage 1 ONE more attempt with that caller code
appended, genuinely resolving the ambiguity instead of leaving a real
vulnerability (or a real false positive) stuck at "inconclusive" for
human review.

Two outcomes, one of them usually requiring no LLM call at all:

* No caller anywhere in the analyzed codebase → the function is never
  invoked from any code path this scan saw. Dead code has no active
  attack surface regardless of what its own body looks like — demoted
  straight to "safe", deterministically, no extra API call.
* Caller(s) found → the model gets ONE more attempt with the caller
  code as CONFIRMED evidence (see ``_RESOLUTION_PROMPT`` below). If
  that resolves the ambiguity, the new verdict replaces the
  inconclusive one. If it's STILL inconclusive, it's left as-is —
  honest, not force-guessed into a verdict.

"No caller" is checked two ways, not one. ``RepositoryIndex.search_usages``
only searches inside OTHER DECLARED FUNCTIONS' bodies — a call sitting in
top-level/module-scope script code (``things.forEach(x => target(x))`` at
a file's top level, a Python ``if __name__ == "__main__":`` block, etc.)
is invisible to it, because the parser never registers top-level code as
a "function" at all. Trusting an empty ``search_usages`` result alone
would silently misreport "no callers found" as "no callers exist" for
that whole class of code — so before declaring dead code, ``_resolve_one``
ALSO greps the unit's own source file (when ``index.repo_path`` was
provided) for the function's name outside every declared function's own
line range. A hit there is fed into the SAME caller-context resolution
call as a tracked-function caller would be, just labeled honestly as
"not a tracked function" so the model knows what kind of evidence it's
looking at. Only when BOTH checks come back empty is it dead code.

Deliberately does NOT reuse ``analyze_unit``/``get_analysis_prompt``
for the re-analysis call. That prompt's own "ANALYZE THIS FUNCTION
ONLY" / "Context (for understanding only — do NOT analyze)" framing —
exactly the framing that makes Stage 1 properly skeptical about
*unproven* origins — was verified (empirically, against real models)
to also suppress it from using a caller that IS proven, i.e. found via
an actual static search of this codebase rather than assumed. This
module's prompt says the opposite: the caller shown is confirmed, not
speculative, so use it.

Generic by design: no repo-specific logic, just "does the index know
who calls this, and if not, is dead code intrinsically safe." Purely
additive — a caller with no ``index`` gets today's behavior exactly.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

# Bounded — mirrors FindingVerifier._find_sibling_call_sites so a
# hot/widely-called function doesn't blow up the re-analysis prompt.
_MAX_CALLERS_SHOWN = 3

# Lines of source shown around an untracked (top-level/module-scope)
# call site, so the model sees enough to judge what's actually being
# passed, not just a bare `target(x)` with no surrounding context.
_UNTRACKED_CALLER_CONTEXT_LINES = 2

_RESOLUTION_PROMPT = """You previously analyzed this function in isolation and correctly said "inconclusive" because you could not see who calls it:

```{language}
{code}
```

A static call-graph search of this codebase (not a claim — an actual search of the parsed source) found the caller(s) below, shown verbatim and unmodified. These ARE the real, confirmed call sites — not a guess about who might call this function.

{callers}

Using this confirmed caller evidence, does the tainted value now have a traceable, attacker-controlled origin? Respond with a JSON object with exactly these keys: "finding" (one of "vulnerable", "safe", "inconclusive"), "reasoning" (string), "attack_vector" (string, or null if not vulnerable). Still say "inconclusive" if the callers shown don't actually settle it — do not force a verdict beyond what the evidence supports."""


def resolve_inconclusive_findings(
    results: list[dict],
    code_by_route: dict,
    binding,
    index: Optional[Any],
    json_corrector=None,
    app_context=None,
) -> list[dict]:
    """Attempt to resolve every "inconclusive" result in ``results`` in place.

    No-op (returns ``results`` unchanged) when ``index`` is None — e.g.
    no ``analyzer_output.json`` path was available to build one.
    """
    if index is None:
        return results

    resolved = 0
    demoted = 0
    for result in results:
        if result.get("finding") != "inconclusive":
            continue
        outcome = _resolve_one(result, binding, code_by_route, index, json_corrector, app_context)
        if outcome == "resolved":
            resolved += 1
        elif outcome == "dead_code":
            demoted += 1

    if resolved or demoted:
        sys.stderr.write(
            f"[Analyze] Inconclusive resolution: {resolved} resolved via caller "
            f"context, {demoted} demoted (no callers found anywhere in this "
            f"codebase — dead code has no active attack surface)\n"
        )
    return results


def _resolve_one(result, binding, code_by_route, index, json_corrector, app_context) -> Optional[str]:
    """Resolve ONE inconclusive result in place.

    Returns "resolved", "dead_code", or None (left inconclusive).
    """
    unit_id = result.get("unit_id")
    if not unit_id:
        return None

    func = index.get_function(unit_id)
    if func:
        function_name = func.get("name")
    else:
        # unit_id didn't match the index's own key format exactly —
        # fall back to the segment after the last ":" (mirrors
        # RepositoryIndex's own "file/path:functionName" convention).
        function_name = unit_id.rsplit(":", 1)[-1] if ":" in unit_id else unit_id
    if not function_name:
        return None

    try:
        usages = index.search_usages(function_name)
    except Exception:
        usages = []

    siblings = [u for u in usages if u.get("id") != unit_id]

    caller_snippets = []
    for u in siblings[:_MAX_CALLERS_SHOWN]:
        caller_id = u.get("id")
        if not caller_id:
            continue
        caller_code = index.get_function_code(caller_id)
        if not caller_code:
            continue
        label = f"// Caller: {caller_id}"
        caller_func = index.get_function(caller_id)
        if caller_func:
            route = caller_func.get("routeMetadata")
            if route:
                label += f" (HTTP {route.get('http_method')} {route.get('http_path')} route handler)"
            elif caller_func.get("isEntryPoint"):
                label += " (entry point)"
        caller_snippets.append(f"{label}\n{caller_code}")

    if len(caller_snippets) < _MAX_CALLERS_SHOWN:
        caller_snippets.extend(
            _find_untracked_source_callers(index, unit_id, function_name)[:_MAX_CALLERS_SHOWN - len(caller_snippets)]
        )

    if not caller_snippets:
        # Neither a tracked-function caller NOR a raw top-level call site
        # anywhere in the unit's own file. Genuinely unreachable from
        # anything this scan parsed.
        result["finding"] = "safe"
        result["verdict"] = "safe"
        result["stage1_inconclusive_resolution"] = "demoted_dead_code"
        return "dead_code"

    route_key = result.get("route_key", unit_id)
    original_code = code_by_route.get(route_key, "")
    if not original_code:
        return None

    prompt = _RESOLUTION_PROMPT.format(
        language="code",  # matches analyze_unit's own generic fence label
        code=original_code,
        callers="\n\n".join(caller_snippets),
    )

    try:
        from utilities.llm import simple_text  # local import: avoid a hard import-time cycle
        from experiment import parse_response

        response = simple_text(binding, prompt, system="You are a security analyst. Be skeptical, but use confirmed evidence you are given rather than ignoring it.")
        new_result = parse_response(response)
    except Exception:
        return None

    new_finding = new_result.get("finding") or (new_result.get("verdict") or "").lower()
    if not new_finding or new_finding in ("inconclusive", "error"):
        return None

    original_unit_id = result.get("unit_id")
    result.update(new_result)
    result["unit_id"] = original_unit_id  # keep the REAL unit_id, not the synthetic one
    result["stage1_inconclusive_resolution"] = "resolved_via_caller_context"
    return "resolved"


def _find_untracked_source_callers(index, unit_id: str, function_name: str) -> list[str]:
    """Grep the unit's own source file for calls to ``function_name``
    that fall OUTSIDE every declared function's line range — i.e. calls
    living in top-level/module-scope script code, which
    ``RepositoryIndex.search_usages`` structurally cannot see (see the
    module docstring). Best-effort: returns ``[]`` whenever the raw
    source isn't available (no ``repo_path``, file missing/unreadable,
    or the function's own file has no line-range metadata) rather than
    raising — this is a supplement to the dead-code check, not a
    replacement for it.
    """
    if getattr(index, "repo_path", None) is None:
        return []

    colon_idx = unit_id.rfind(":")
    file_path = unit_id[:colon_idx] if colon_idx > 0 else None
    if not file_path:
        return []

    try:
        source = (index.repo_path / file_path).read_text(encoding="utf-8")
    except Exception:
        return []
    lines = source.splitlines()

    covered_ranges = []
    for func_id in index.by_file.get(file_path, []):
        func_data = index.functions.get(func_id, {})
        start, end = func_data.get("startLine"), func_data.get("endLine")
        if isinstance(start, int) and isinstance(end, int):
            covered_ranges.append((start, end))

    def _is_covered(line_no: int) -> bool:
        return any(start <= line_no <= end for start, end in covered_ranges)

    call_pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    snippets = []
    for i, line in enumerate(lines):
        line_no = i + 1  # analyzer's startLine/endLine are 1-indexed
        if _is_covered(line_no) or not call_pattern.search(line):
            continue
        lo = max(0, i - _UNTRACKED_CALLER_CONTEXT_LINES)
        hi = min(len(lines), i + _UNTRACKED_CALLER_CONTEXT_LINES + 1)
        context = "\n".join(lines[lo:hi])
        snippets.append(
            f"// Caller (top-level/module-scope code, not a tracked function; "
            f"{file_path} line {line_no}):\n{context}"
        )
    return snippets
