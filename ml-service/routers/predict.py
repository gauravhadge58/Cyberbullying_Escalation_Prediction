"""
/predict endpoint — predicts bullying + escalation for new messages,
then fires the LangChain moderation layer asynchronously as a background task.

Pipeline:
  Chat Messages
    → BERT Toxicity Detection (detection.predict)
    → BiLSTM + Random Forest Escalation (escalation.predict_conversation)
    → [Immediate HTTP response returned here]
    → LangChain Explanation & Recommendation (background task → WebSocket broadcast)
"""
import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from models import detection, escalation

logger = logging.getLogger(__name__)
router = APIRouter()


class Message(BaseModel):
    id: str
    conversation_id: str
    user_id: Optional[str] = "unknown"
    timestamp: Optional[str] = None
    message: str


class PredictRequest(BaseModel):
    messages: List[Message]


# ─────────────────────────────────────────────────────────────────────────────
# Background task: LangChain moderation
# ─────────────────────────────────────────────────────────────────────────────

async def _run_moderation_background(
    conv_id: str,
    features: dict,
    messages_dicts: list,
    escalation_result: dict,
):
    """
    Called as a FastAPI BackgroundTask after the HTTP response is sent.
    Runs the LangChain explanation pipeline and broadcasts result via the
    backend Node.js WebSocket infrastructure (POST /api/ml-events).
    """
    langchain_enabled = os.getenv("LANGCHAIN_ENABLED", "true").lower() != "false"
    if not langchain_enabled:
        return

    try:
        from langchain_moderator.moderator import get_moderator
        moderator = get_moderator()
        result = await moderator.generate_explanation_async(
            conv_id=conv_id,
            features=features,
            messages=messages_dicts,
            escalation_result=escalation_result,
        )

        # POST the result to the Node.js backend which broadcasts it via WebSocket
        # BACKEND_URL env var is how the ML service knows where the backend is
        backend_url = os.getenv("BACKEND_URL", "http://localhost:5000")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{backend_url}/api/ml-events",
                    json={
                        "type": "MODERATION_UPDATE",
                        "data": {
                            "conversation_id": conv_id,
                            "moderation": result,
                        },
                    },
                )
            logger.info(
                "Moderation update sent to backend for conv %s (action=%s)",
                conv_id, result.get("suggested_action"),
            )
        except Exception as ws_exc:
            logger.warning(
                "Could not post MODERATION_UPDATE to backend for conv %s: %s "
                "(Set BACKEND_URL in ml-service/.env — default: http://localhost:5000)",
                conv_id, ws_exc,
            )

    except Exception as exc:
        logger.error("Background moderation task failed for conv %s: %s", conv_id, exc)



# ─────────────────────────────────────────────────────────────────────────────
# /predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(request_data: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict cyberbullying + escalation for submitted messages.
    Messages may belong to multiple conversations.

    Returns per-message detection results and per-conversation escalation.
    After the response is sent, fires LangChain moderation as a background task
    and broadcasts the result via WebSocket (MODERATION_UPDATE event).
    """
    try:
        if not request_data.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Group messages by conversation_id
        conversations: dict[str, list] = {}
        for msg in request_data.messages:
            conversations.setdefault(msg.conversation_id, []).append(msg)

        # Run BERT detection on all messages at once
        texts = [m.message for m in request_data.messages]
        detection_results = detection.predict(texts)

        # Map per-message detection results back
        msg_results = {}
        for msg, det in zip(request_data.messages, detection_results):
            msg_results[msg.id] = {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "user_id": msg.user_id,
                "timestamp": msg.timestamp or datetime.utcnow().isoformat(),
                "message": msg.message,
                "is_bullying": det["is_bullying"],
                "confidence": det["confidence"],
                "toxicity_score": det["toxicity_score"],
            }

        # Per-conversation BiLSTM + Random Forest escalation
        conv_results = {}
        conv_features = {}   # Store features per conv for LangChain
        conv_messages = {}   # Store enriched message dicts for LangChain

        for conv_id, msgs in conversations.items():
            enriched = []
            for m in msgs:
                d = msg_results[m.id]
                enriched.append({
                    "id": m.id,
                    "conversation_id": conv_id,
                    "user_id": m.user_id,
                    "timestamp": m.timestamp,
                    "message": m.message,
                    "toxicity_score": d["toxicity_score"],
                    "is_bullying": d["is_bullying"],
                })
            esc = escalation.predict_conversation(enriched)
            conv_results[conv_id] = {
                "conversation_id": conv_id,
                "escalation_level": esc["escalation_level"],
                "escalation_score": esc["escalation_score"],
                "features": esc["features"],
                "message_count": len(msgs),
            }
            conv_features[conv_id] = esc["features"]
            conv_messages[conv_id] = enriched

        # ── Schedule LangChain moderation as background task for each conversation ──
        for conv_id, esc_result in conv_results.items():
            background_tasks.add_task(
                _run_moderation_background,
                conv_id=conv_id,
                features=conv_features[conv_id],
                messages_dicts=conv_messages[conv_id],
                escalation_result=esc_result,
            )

        return {
            "messages": list(msg_results.values()),
            "conversations": list(conv_results.values()),
        }

    except Exception as e:
        import traceback
        error_msg = f"Predict Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
