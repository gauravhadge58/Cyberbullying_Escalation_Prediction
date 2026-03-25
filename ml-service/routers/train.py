"""
/train endpoint — accepts CSV upload, trains detection + escalation models.
"""
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from models import detection, escalation

router = APIRouter()


@router.post("/train")
async def train_models(file: UploadFile = File(...)):
    """
    Upload a labelled CSV file to train both models.

    Expected CSV columns:
        id, conversation_id, user_id, timestamp, message, label
        label: 1 = bullying, 0 = non-bullying
    """
    if not file.filename.endswith((".csv", ".CSV")):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    required = {"message"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Ensure label column exists (default 0 if absent)
    if "label" not in df.columns:
        df["label"] = 0

    # Train detection model
    detection_metrics = detection.train(df)

    # Run detection predictions to enrich escalation training data
    preds = detection.predict(df["message"].fillna("").tolist())
    df["toxicity_score"] = [p["toxicity_score"] for p in preds]
    df["is_bullying"] = [p["is_bullying"] for p in preds]

    # Train escalation model (needs conversation_id + timestamp)
    escalation_metrics = {}
    if "conversation_id" in df.columns:
        escalation_metrics = escalation.train(df)

    return {
        "status": "Training complete",
        "detection": detection_metrics,
        "escalation": escalation_metrics,
    }
