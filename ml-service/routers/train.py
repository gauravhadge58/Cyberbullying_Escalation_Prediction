"""
/train endpoint — accepts CSV upload, trains detection + escalation + LSTM models.
Registers trained models in the model registry.
"""
import io
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException

from models import detection, escalation
from models import lstm_model as lstm
import model_registry as registry

router = APIRouter()

# Global state for streaming progress to the UI
_training_progress = {
    "status": "idle",
    "phase": "",
    "current": 0,
    "total": 0,
    "message": ""
}


@router.post("/train")
def train_models(file: UploadFile = File(...)):
    """
    Upload a labelled CSV file to train all models.

    Expected CSV columns:
        id, conversation_id, user_id, timestamp, message, label
        label: 1 = bullying, 0 = non-bullying
    """
    if not file.filename.endswith((".csv", ".CSV")):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = file.file.read()
    
    global _training_progress
    _training_progress = {
        "status": "loading",
        "phase": "init",
        "current": 0,
        "total": 0,
        "message": "Initializing training pipeline..."
    }
    
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        _training_progress.update({"status": "error", "message": f"Failed to parse CSV: {e}"})
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    required = {"message"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Ensure label column exists (default 0 if absent)
    if "label" not in df.columns:
        df["label"] = 0

    # Train detection model
    print(f"🚀 [1/3] Starting Detection phase for {len(df)} rows...")
    detection_metrics = detection.train(df)

    # Run detection predictions to enrich escalation training data
    print(f"⏳ Scoring {len(df)} messages with BERT (this uses the GPU and takes the longest)...")
    _training_progress.update({
        "phase": "bert",
        "message": "Extracting hyper-dimensional semantic features via Transformer...",
        "current": 0,
        "total": len(df)
    })
    
    def bert_progress(current, total):
        _training_progress["current"] = current
        _training_progress["total"] = total

    preds = detection.predict(df["message"].fillna("").tolist(), progress_cb=bert_progress)
    df["toxicity_score"] = [p["toxicity_score"] for p in preds]
    df["is_bullying"] = [p["is_bullying"] for p in preds]
    print(f"✅ BERT scoring complete!")

    # Shared model ID (timestamp) for this training run
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Train escalation model (Random Forest)
    print(f"🚀 [2/3] Training Random Forest escalation model...")
    _training_progress.update({
        "phase": "rf",
        "message": "Building ensemble decision boundaries...",
        "current": 0,
        "total": 100
    })
    
    escalation_metrics = {}
    if "conversation_id" in df.columns:
        escalation_metrics = escalation.train(df)
        print(f"✅ Random Forest complete!")
        _training_progress["current"] = 100

    # Train LSTM model
    print(f"🚀 [3/3] Training LSTM sequential model on GPU...")
    _training_progress.update({
        "phase": "lstm",
        "message": "Propagating temporal escalation gradients on GPU...",
        "current": 0,
        "total": 20
    })

    def lstm_progress(current, total):
        _training_progress["current"] = current
        _training_progress["total"] = total
        
    lstm_metrics = {}
    if "conversation_id" in df.columns:
        lstm_metrics = lstm.train(df, model_id=model_id, progress_cb=lstm_progress)

    # Register in model registry
    print(f"💾 Registering new models with ID: {model_id}...")
    rf_filename = escalation_metrics.get("rf_filename")
    enc_filename = escalation_metrics.get("enc_filename")
    lstm_filename = lstm_metrics.get("lstm_filename")

    entry = registry.register_new_model(
        model_id=model_id,
        rf_filename=rf_filename,
        encoder_filename=enc_filename,
        lstm_filename=lstm_filename,
        metrics={
            "num_conversations": lstm_metrics.get("num_conversations"),
            "lstm_accuracy": lstm_metrics.get("training_accuracy"),
            "lstm_final_loss": lstm_metrics.get("final_loss"),
            "rf_classes": escalation_metrics.get("classes"),
        },
        set_active=True,
    )
    print(f"🎉 Pipeline Complete! Active model is now {model_id}.")

    _training_progress.update({
        "status": "success",
        "phase": "complete",
        "message": "Pipeline Complete!"
    })

    return {
        "status": "Training complete",
        "model_id": model_id,
        "detection": detection_metrics,
        "escalation": escalation_metrics,
        "lstm": lstm_metrics,
    }


# ─────────────────────────────────────────────
# Model Registry & Progress Endpoints
# ─────────────────────────────────────────────

@router.get("/train/progress")
def get_training_progress():
    """Returns the current training progress state for the UI pipeline."""
    global _training_progress
    return _training_progress


@router.get("/models")
def list_models():
    """Return all registered models and the active model ID."""
    reg = registry.get_registry()
    return {
        "active_id": reg.get("active_id"),
        "models": reg.get("models", []),
    }


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str):
    """Set a model as the active model for predictions."""
    success = registry.activate_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return {"status": "ok", "active_id": model_id}


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Remove a model from the registry."""
    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    reg = registry.get_registry()
    return {"status": "deleted", "active_id": reg.get("active_id")}
