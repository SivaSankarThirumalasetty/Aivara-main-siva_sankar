import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight import cache as insight_cache


def test_cross_user_cache_isolation_per_session():
    """Verify that view-level insights set for Session A cannot be retrieved by Session B."""
    session_a = "user_session_111"
    session_b = "user_session_222"

    insight_a = {
        "headline": "Company A Revenue grew 40%.",
        "driver_explanation": "Enterprise segment drove the gain.",
        "suggested_action": "Expand Enterprise team.",
    }
    insight_b = {
        "headline": "Company B Revenue declined 10%.",
        "driver_explanation": "Retail segment contracted.",
        "suggested_action": "Audit retail pipeline.",
    }

    # Set insight for Session A on Executive view
    insight_cache.set_for_view("Executive", insight_a, session_id=session_a)

    # Set insight for Session B on Executive view
    insight_cache.set_for_view("Executive", insight_b, session_id=session_b)

    # Session A must retrieve Company A's insight
    retrieved_a = insight_cache.get_for_view("Executive", session_id=session_a)
    assert retrieved_a is not None
    assert retrieved_a["headline"] == "Company A Revenue grew 40%."

    # Session B must retrieve Company B's insight and NOT Company A's
    retrieved_b = insight_cache.get_for_view("Executive", session_id=session_b)
    assert retrieved_b is not None
    assert retrieved_b["headline"] == "Company B Revenue declined 10%."

    # Unknown session C must retrieve None, NOT Session A or B's data
    retrieved_c = insight_cache.get_for_view("Executive", session_id="user_session_333")
    assert retrieved_c is None

    # Clearing Session A must not affect Session B
    insight_cache.clear_view_cache(session_id=session_a)
    assert insight_cache.get_for_view("Executive", session_id=session_a) is None
    assert insight_cache.get_for_view("Executive", session_id=session_b) == insight_b
