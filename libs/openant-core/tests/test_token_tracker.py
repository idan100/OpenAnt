"""Tests for TokenTracker."""
from utilities.llm_client import TokenTracker, MODEL_PRICING


class TestTokenTracker:
    def test_initial_state(self):
        tracker = TokenTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.calls == []

    def test_record_call_known_model(self):
        tracker = TokenTracker()
        result = tracker.record_call("claude-sonnet-4-20250514", 1000, 500)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        # Sonnet: $3/M input, $15/M output
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert result["cost_usd"] == round(expected_cost, 6)

    def test_record_call_unknown_model_reports_zero_cost(self):
        # Issue #65: unknown models report $0 with a one-time warning
        # rather than silently estimating at Sonnet rates. Token counts
        # are still recorded; only the cost is zeroed.
        tracker = TokenTracker()
        result = tracker.record_call("some-future-model", 100, 50)
        assert result["cost_usd"] == 0.0
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_cumulative_tracking(self):
        tracker = TokenTracker()
        tracker.record_call("claude-sonnet-4-20250514", 1000, 500)
        tracker.record_call("claude-sonnet-4-20250514", 2000, 1000)

        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.total_tokens == 4500
        assert len(tracker.calls) == 2

    def test_reset(self):
        tracker = TokenTracker()
        tracker.record_call("claude-sonnet-4-20250514", 1000, 500)
        tracker.reset()

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.calls == []

    def test_get_summary_includes_calls(self):
        tracker = TokenTracker()
        tracker.record_call("claude-sonnet-4-20250514", 100, 50)
        summary = tracker.get_summary()

        assert summary["total_calls"] == 1
        assert "calls" in summary
        assert len(summary["calls"]) == 1

    def test_get_totals_excludes_calls(self):
        tracker = TokenTracker()
        tracker.record_call("claude-sonnet-4-20250514", 100, 50)
        totals = tracker.get_totals()

        assert totals["total_calls"] == 1
        assert "calls" not in totals

    def test_opus_pricing(self):
        tracker = TokenTracker()
        result = tracker.record_call("claude-opus-4-20250514", 1_000_000, 1_000_000)
        # Opus: $15/M input, $75/M output
        assert result["cost_usd"] == 90.0

    def test_record_call_defaults_cache_tokens_to_zero(self):
        tracker = TokenTracker()
        result = tracker.record_call("claude-sonnet-4-20250514", 100, 50)
        assert result["cache_creation_input_tokens"] == 0
        assert result["cache_read_input_tokens"] == 0
        assert result["cache_savings_usd"] == 0.0

    def test_cache_read_billed_at_discount(self):
        tracker = TokenTracker()
        result = tracker.record_call(
            "claude-sonnet-4-20250514", 0, 0, cache_read_input_tokens=1_000_000
        )
        # Sonnet input $3/M * 0.1 (cache-read multiplier) = $0.30
        assert result["cost_usd"] == 0.30
        # Savings vs. paying full input price for those tokens: $3 - $0.30
        assert result["cache_savings_usd"] == 2.70

    def test_cache_write_billed_at_premium(self):
        tracker = TokenTracker()
        result = tracker.record_call(
            "claude-sonnet-4-20250514", 0, 0, cache_creation_input_tokens=1_000_000
        )
        # Sonnet input $3/M * 1.25 (cache-write multiplier) = $3.75
        assert result["cost_usd"] == 3.75

    def test_cumulative_cache_metrics_in_totals(self):
        tracker = TokenTracker()
        tracker.record_call(
            "claude-sonnet-4-20250514", 100, 50,
            cache_creation_input_tokens=1000, cache_read_input_tokens=0,
        )
        tracker.record_call(
            "claude-sonnet-4-20250514", 100, 50,
            cache_creation_input_tokens=0, cache_read_input_tokens=9000,
        )
        totals = tracker.get_totals()
        assert totals["cache_creation_input_tokens"] == 1000
        assert totals["cache_read_input_tokens"] == 9000
        # hit rate = cache_read / (input + cache_creation + cache_read)
        #          = 9000 / (200 + 1000 + 9000)
        assert totals["cache_hit_rate"] == round(9000 / 10200, 4)
        assert totals["cache_savings_usd"] > 0

    def test_unknown_model_zeroes_cache_savings_too(self):
        tracker = TokenTracker()
        result = tracker.record_call(
            "some-future-model", 100, 50, cache_read_input_tokens=1000
        )
        assert result["cost_usd"] == 0.0
        assert result["cache_savings_usd"] == 0.0
