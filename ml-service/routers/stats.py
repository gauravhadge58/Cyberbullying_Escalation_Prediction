"""
/stats endpoint — returns aggregate analytics data.
"""
from fastapi import APIRouter

router = APIRouter()

# In-memory analytics store
_stats_cache: dict = {}


def update_stats(stats: dict):
    """Update the stats cache from predict/train calls."""
    global _stats_cache
    _stats_cache.update(stats)


@router.get("/stats")
def get_stats():
    """Return aggregate analytics: message counts, bullying %, escalation distribution."""
    return _stats_cache if _stats_cache else {
        "total_messages": 0,
        "bullying_count": 0,
        "non_bullying_count": 0,
        "bullying_percentage": 0.0,
        "escalation_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
        "high_escalation_conversations": [],
        "toxicity_over_time": [],
    }
