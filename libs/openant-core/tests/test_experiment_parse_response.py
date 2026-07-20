"""Tests for experiment.py's parse_response() guarding against JSON that
parses successfully but isn't an object (a bare list/string/number) —
e.g. a model wrapping its answer as ``[{"finding": "safe"}]``. Before
this fix, ``_normalize_result`` would then crash with a TypeError on
its first item-assignment (``result["verdict"] = ...``) since a list
doesn't support that. This treats it exactly like a JSON decode
failure: falls through to the find-the-{-and-} recovery, then the
standard error dict — never crashes.
"""

from __future__ import annotations

from experiment import parse_response


def test_normal_object_response_parses_fine():
    result = parse_response('{"finding": "safe", "reasoning": "no issue"}')
    assert result["verdict"] == "SAFE"


def test_bare_list_response_does_not_crash():
    # A list wrapping one object doesn't crash -- and the find-the-{-and-}
    # recovery is smart enough to pull the embedded object straight out
    # rather than needing to fall all the way to an error. The key
    # property under test is simply: no TypeError from _normalize_result
    # trying to item-assign into a list.
    result = parse_response('[{"finding": "safe"}]')
    assert result["verdict"] == "SAFE"


def test_bare_string_response_does_not_crash():
    result = parse_response('"just a string, not an object"')
    assert result["verdict"] == "ERROR"


def test_bare_number_response_does_not_crash():
    result = parse_response("42")
    assert result["verdict"] == "ERROR"


def test_object_embedded_in_surrounding_text_still_recovers():
    # The find-the-{-and-} fallback should still work for the ordinary
    # "model added prose around the JSON" case.
    result = parse_response('Here is my answer:\n{"finding": "vulnerable"}\nThanks!')
    assert result["verdict"] == "VULNERABLE"


def test_object_embedded_inside_a_list_in_prose_still_recovers():
    # The find-the-{-and-} recovery looks for ANY {...} span, including
    # one nested inside a list -- still recovers cleanly, still never
    # crashes trying to item-assign into the outer list.
    result = parse_response('Here is my answer:\n[{"finding": "vulnerable"}]\nThanks!')
    assert result["verdict"] == "VULNERABLE"


def test_markdown_fenced_object_still_parses():
    result = parse_response('```json\n{"finding": "safe"}\n```')
    assert result["verdict"] == "SAFE"
