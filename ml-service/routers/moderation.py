"""
/moderation/* endpoints — LangChain-powered moderation reasoning layer.

Endpoints:
  POST /moderation/explain          — On-demand explanation for a conversation
  GET  /moderation/summary/{conv_id} — Short conversation summary
  GET  /moderation/health           — LLM provider health check
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from langchain_moderator.moderator import get_moderator

router = APIRouter(prefix="/moderation", tags=["Moderation"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class MessagePayload(BaseModel):
    id: Optional[str] = None
    conversation_id: str
    user_id: Optional[str] = "unknown"
    message: str
    timestamp: Optional[str] = None


class ExplainRequest(BaseModel):
    conversation_id: str
    escalation_level: str                   # LOW | MEDIUM | HIGH
    escalation_score: Optional[float] = 0
    features: Dict[str, Any] = {}
    messages: List[MessagePayload] = []


class ExplainResponse(BaseModel):
    conv_id: str
    explanation: str
    reasoning: str
    suggested_action: str
    action_label: str
    summary: str
    provider: str


class SummaryResponse(BaseModel):
    conv_id: str
    summary: str
    provider: str


class HealthResponse(BaseModel):
    status: str
    enabled: bool
    provider: str
    model: str
    chain_ready: bool


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def moderation_health():
    """
    Check LangChain layer health — verifies LLM provider is reachable.
    Does NOT make an LLM call; only checks initialisation status.
    """
    moderator = get_moderator()
    return moderator.is_healthy()


@router.post("/explain", response_model=ExplainResponse)
async def explain_conversation(request: ExplainRequest):
    """
    Generate an AI moderation explanation for a conversation.

    Accepts the full escalation result + message list, runs through the
    LangChain reasoning pipeline (PromptTemplate + LLM + RAG), and returns
    a structured moderation report with explanation, reasoning, and action.

    This is the **synchronous** on-demand endpoint (e.g., clicking
    'Request AI Analysis' in the Monitor dashboard). For real-time use
    during prediction, the background task in /predict handles it.
    """
    moderator = get_moderator()

    messages_as_dicts = [m.model_dump() for m in request.messages]
    escalation_result = {
        "escalation_level": request.escalation_level,
        "escalation_score": request.escalation_score,
        "features": request.features,
    }

    try:
        result = await moderator.generate_explanation_async(
            conv_id=request.conversation_id,
            features=request.features,
            messages=messages_as_dicts,
            escalation_result=escalation_result,
        )
        return ExplainResponse(**result)
    except Exception as exc:
        logger.error("Moderation explain error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Moderation explanation failed: {exc}")


@router.get("/summary/{conv_id}", response_model=SummaryResponse)
async def conversation_summary(conv_id: str):
    """
    Return a short natural-language summary of a conversation's escalation
    pattern based on its stored memory context.

    Primarily used by the Monitor dashboard to show a quick overview without
    re-submitting all messages.
    """
    moderator = get_moderator()
    memory = moderator.memory.get(conv_id)

    if not memory:
        return SummaryResponse(
            conv_id=conv_id,
            summary="No memory context found for this conversation. Submit a prediction first.",
            provider="none",
        )

    # Reconstruct minimal feature dict from memory
    last_snapshot = {}
    if memory.get("escalation_history"):
        last_snapshot = memory["escalation_history"][-1].get("features_snapshot", {})
    level = memory["escalation_history"][-1]["level"] if memory.get("escalation_history") else "UNKNOWN"

    # Build fake messages list from stored texts for summarisation
    messages = [{"message": t} for t in memory.get("recent_messages", [])]

    try:
        summary_text = await moderator.summarize_conversation_async(
            conv_id=conv_id,
            messages=messages,
            features=last_snapshot,
            level=level,
        )
        return SummaryResponse(
            conv_id=conv_id,
            summary=summary_text,
            provider=moderator.is_healthy().get("provider", "groq"),
        )
    except Exception as exc:
        logger.error("Moderation summary error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc}")
