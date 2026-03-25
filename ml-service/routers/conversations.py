"""
/conversations endpoint — returns recently analyzed conversations (in-memory cache).
In production this should be fetched from MongoDB via the Node.js backend.
"""
from fastapi import APIRouter
from typing import List

router = APIRouter()

# In-memory store (populated by /predict calls in production)
_conversation_cache: List[dict] = []


def update_cache(conversations: List[dict]):
    """Called by predict router to update the cache."""
    global _conversation_cache
    _conversation_cache = conversations


@router.get("/conversations")
def get_conversations():
    """Return all analyzed conversations from the cache."""
    return {"conversations": _conversation_cache, "count": len(_conversation_cache)}
