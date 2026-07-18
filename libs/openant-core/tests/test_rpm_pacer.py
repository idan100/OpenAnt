"""Tests for RpmPacer and the (provider, model) RPM-pacer registry."""

from __future__ import annotations

import threading
from unittest.mock import patch

import utilities.rate_limiter as rl
from utilities.llm.providers._ratelimit import wait_for_rate_limit


def _isolated_pacer(rpm_limit: float, monkeypatch, start_t: float = 1000.0):
    """Fresh RpmPacer with mocked, controllable time.sleep/monotonic."""
    pacer = rl.RpmPacer(rpm_limit)
    clock = {"t": start_t}
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rl.random, "uniform", lambda a, b: 0.0)  # no jitter

    def fake_sleep(d):
        clock["t"] += d

    monkeypatch.setattr(rl.time, "sleep", fake_sleep)
    return pacer, clock


class TestRpmPacer:
    def test_allows_up_to_limit_without_blocking(self, monkeypatch):
        pacer, clock = _isolated_pacer(3, monkeypatch)
        for _ in range(3):
            assert pacer.wait_for_slot() == 0.0
        assert clock["t"] == 1000.0  # no sleeping happened

    def test_blocks_past_limit_until_oldest_ages_out(self, monkeypatch):
        pacer, clock = _isolated_pacer(2, monkeypatch)
        pacer.wait_for_slot()  # t=1000
        clock["t"] += 10
        pacer.wait_for_slot()  # t=1010, both slots used
        clock["t"] += 5  # t=1015 - oldest (t=1000) ages out at t=1060

        waited = pacer.wait_for_slot()
        assert waited > 0.0
        assert clock["t"] >= 1060.0  # slept until the 60s window cleared

    def test_reset_clears_history(self, monkeypatch):
        pacer, clock = _isolated_pacer(1, monkeypatch)
        pacer.wait_for_slot()
        pacer.reset()
        # Immediately after reset, a fresh slot is available with no wait.
        assert pacer.wait_for_slot() == 0.0
        assert clock["t"] == 1000.0

    def test_thread_safety_smoke(self):
        # Not mocking time here — a real, fast smoke test that many
        # threads sharing one pacer never oversubscribe the limit and
        # never crash/deadlock.
        pacer = rl.RpmPacer(1000)  # high enough that none should block
        results = []
        lock = threading.Lock()

        def worker():
            waited = pacer.wait_for_slot()
            with lock:
                results.append(waited)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(results) == 20


class TestRpmPacerRegistry:
    def setup_method(self):
        rl.reset_rpm_pacers()

    def teardown_method(self):
        rl.reset_rpm_pacers()

    def test_no_pacer_configured_returns_none(self):
        assert rl.get_rpm_pacer("google", "gemini-x") is None

    def test_configure_creates_pacer_keyed_by_provider_and_model(self):
        rl.configure_rpm_limit("google", "gemini-a", 5)
        rl.configure_rpm_limit("google", "gemini-b", 10)
        a = rl.get_rpm_pacer("google", "gemini-a")
        b = rl.get_rpm_pacer("google", "gemini-b")
        assert a is not None and b is not None
        assert a is not b
        assert a._limit == 5
        assert b._limit == 10

    def test_configure_none_clears_pacer(self):
        rl.configure_rpm_limit("google", "gemini-a", 5)
        assert rl.get_rpm_pacer("google", "gemini-a") is not None
        rl.configure_rpm_limit("google", "gemini-a", None)
        assert rl.get_rpm_pacer("google", "gemini-a") is None

    def test_same_provider_different_model_independent(self):
        # The whole point: one "google" provider spanning models with
        # very different RPM ceilings must not share one pacer.
        rl.configure_rpm_limit("google", "gemini-3.1-flash-lite", 15)
        rl.configure_rpm_limit("google", "gemini-3.5-flash", 5)
        assert rl.get_rpm_pacer("google", "gemini-3.1-flash-lite")._limit == 15
        assert rl.get_rpm_pacer("google", "gemini-3.5-flash")._limit == 5


class TestWaitForRateLimitConsultsPacer:
    def setup_method(self):
        rl.reset_rate_limiter()
        rl.reset_rpm_pacers()

    def teardown_method(self):
        rl.reset_rate_limiter()
        rl.reset_rpm_pacers()

    def test_no_model_skips_pacer_entirely(self):
        # Back-compat: a caller with no model in scope still works,
        # backoff-only behavior — must not raise on missing model.
        wait_for_rate_limit("anthropic")

    def test_no_pacer_configured_for_model_is_a_no_op(self):
        wait_for_rate_limit("google", "gemini-unconfigured")  # must not raise

    def test_configured_pacer_is_consulted(self):
        rl.configure_rpm_limit("google", "gemini-x", 5)
        pacer = rl.get_rpm_pacer("google", "gemini-x")
        with patch.object(pacer, "wait_for_slot", wraps=pacer.wait_for_slot) as spy:
            wait_for_rate_limit("google", "gemini-x")
        spy.assert_called_once()
