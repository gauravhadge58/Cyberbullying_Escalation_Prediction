"""
/predict endpoint — predicts bullying + escalation for new messages.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from models import detection, escalation

router = APIRouter()


class Message(BaseModel):
    id: str
    conversation_id: str
    user_id: Optional[str] = "unknown"
    timestamp: Optional[str] = None
    message: str


class PredictRequest(BaseModel):
    messages: List[Message]


@router.post("/predict")
def predict(request: PredictRequest):
    """
    Predict cyberbullying + escalation for submitted messages.
    Messages may belong to multiple conversations.
    
    Returns per-message detection results and per-conversation escalation.
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Group messages by conversation_id
        conversations: dict[str, list] = {}
        for msg in request.messages:
            conversations.setdefault(msg.conversation_id, []).append(msg)

        # Run detection on all messages at once
        texts = [m.message for m in request.messages]
        detection_results = detection.predict(texts)

        # Map per-message results back
        msg_results = {}
        for msg, det in zip(request.messages, detection_results):
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

        # Per-conversation escalation
        conv_results = {}
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

        return {
            "messages": list(msg_results.values()),
            "conversations": list(conv_results.values()),
        }
    except Exception as e:
        import traceback
        error_msg = f"Predict Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
