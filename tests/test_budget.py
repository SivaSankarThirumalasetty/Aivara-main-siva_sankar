import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight import budget


def test_budget_allows_calls_under_ceiling():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "budget_state.json")
        assert budget.can_make_call(path=path, daily_budget=3)
        budget.record_call(path=path)
        budget.record_call(path=path)
        assert budget.can_make_call(path=path, daily_budget=3)


def test_budget_blocks_calls_past_ceiling():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "budget_state.json")
        for _ in range(3):
            budget.record_call(path=path)
        assert not budget.can_make_call(path=path, daily_budget=3)
        status = budget.get_status(path=path, daily_budget=3)
        assert status.exhausted
        assert status.remaining == 0


def test_budget_state_persists_across_calls_to_get_status():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "budget_state.json")
        budget.record_call(path=path)
        budget.record_call(path=path)
        status = budget.get_status(path=path, daily_budget=30)
        assert status.calls_made_today == 2
        assert status.remaining == 28


def test_estimate_tokens_returns_positive_int():
    assert budget.estimate_tokens("hello world, this is a short facts packet") > 0
