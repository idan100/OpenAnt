"""Regression test — write_json is non-atomic (truncate-then-write).

write_json opens the target in "w" (truncating it to 0 bytes) and then streams json.dump. If the process is
killed mid-serialization (SIGKILL / OOM / power loss), the target is left partial/empty and the previous good
content is lost → the next read_json raises JSONDecodeError. Fix: write to a temp file in the same directory and
os.replace it onto the target (atomic rename), so an interrupted write never clobbers the prior version.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # libs/openant-core

import utilities.file_io as fio  # noqa: E402
from utilities.file_io import read_json, write_json  # noqa: E402


def test_write_json_atomic_preserves_original_on_crash(tmp_path, monkeypatch):
    """An interrupted write must leave the prior good file intact (atomic replace)."""
    p = tmp_path / "checkpoint.json"
    write_json(p, {"v": 1})
    assert read_json(p) == {"v": 1}

    def boom(*a, **k):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(fio.json, "dump", boom)
    with pytest.raises(RuntimeError):
        write_json(p, {"v": 2})

    # Post-fix: the original is untouched. Pre-fix: p was truncated -> empty -> this raises/!= {"v":1}.
    assert read_json(p) == {"v": 1}, "interrupted write clobbered the prior checkpoint"
    leftovers = list(tmp_path.glob(".tmp*"))
    assert not leftovers, f"temp file leaked after failed write: {leftovers}"


def test_write_json_normal_roundtrip(tmp_path):
    """Guard: the normal write path still round-trips correctly."""
    p = tmp_path / "x.json"
    write_json(p, {"a": [1, 2, 3], "b": "héllo"})
    assert read_json(p) == {"a": [1, 2, 3], "b": "héllo"}


# ---------------------------------------------------------------------------
# Windows WinError 5 (PermissionError) on os.replace() — a concurrent
# reader/scanner briefly holding the target open. Retries a few times
# with backoff before giving up, Windows-only (POSIX rename doesn't
# have this failure mode).
# ---------------------------------------------------------------------------


def test_replace_retries_and_succeeds_on_windows(monkeypatch):
    monkeypatch.setattr(fio.sys, "platform", "win32")
    monkeypatch.setattr(fio.time, "sleep", lambda _: None)  # keep the test instant
    calls = {"n": 0}

    def flaky_replace(tmp, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("WinError 5: Access is denied")

    monkeypatch.setattr(fio.os, "replace", flaky_replace)
    fio._replace_with_windows_retry("tmp", "target")
    assert calls["n"] == 3, "must retry through the transient PermissionError and then succeed"


def test_replace_gives_up_after_exhausting_retries_on_windows(monkeypatch):
    monkeypatch.setattr(fio.sys, "platform", "win32")
    monkeypatch.setattr(fio.time, "sleep", lambda _: None)

    def always_fails(tmp, target):
        raise PermissionError("WinError 5: Access is denied")

    monkeypatch.setattr(fio.os, "replace", always_fails)
    with pytest.raises(PermissionError):
        fio._replace_with_windows_retry("tmp", "target")


def test_replace_does_not_retry_on_non_windows(monkeypatch):
    monkeypatch.setattr(fio.sys, "platform", "linux")
    calls = {"n": 0}

    def always_fails(tmp, target):
        calls["n"] += 1
        raise PermissionError("some other permission issue")

    monkeypatch.setattr(fio.os, "replace", always_fails)
    with pytest.raises(PermissionError):
        fio._replace_with_windows_retry("tmp", "target")
    assert calls["n"] == 1, "POSIX rename doesn't have this failure mode -- must not retry"


def test_write_json_end_to_end_survives_transient_windows_permission_error(tmp_path, monkeypatch):
    """The full write_json() path, not just the retry helper in isolation."""
    monkeypatch.setattr(fio.sys, "platform", "win32")
    monkeypatch.setattr(fio.time, "sleep", lambda _: None)
    p = tmp_path / "checkpoint.json"
    real_replace = fio.os.replace
    calls = {"n": 0}

    def flaky_replace(tmp, target):
        calls["n"] += 1
        if calls["n"] < 2:
            raise PermissionError("WinError 5: Access is denied")
        real_replace(tmp, target)

    monkeypatch.setattr(fio.os, "replace", flaky_replace)
    write_json(p, {"v": 1})
    assert read_json(p) == {"v": 1}
    assert calls["n"] == 2
