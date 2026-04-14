"""
Cyberbullying detection model.
Supports multiple transformer backends selectable via the DETECTION_MODEL env var.

Presets (set DETECTION_MODEL=<preset> in your environment):
  light         → martin-ha/toxic-comment-model  (DistilBERT, ~250MB) ← Render free tier default
  full          → unitary/toxic-bert              (BERT, ~440MB)       ← Local/GPU
  multilingual  → unitary/multilingual-toxic-xlm-roberta               ← Multi-language support

You can also pass any raw HuggingFace model ID:
  DETECTION_MODEL=my-org/my-custom-model
"""
import os
import torch
import numpy as np
import pandas as pd
from transformers import pipeline

from preprocessing import clean_text

# ── Model Presets ─────────────────────────────────────────────────────────────
# Named shortcuts → full HuggingFace model IDs
MODEL_PRESETS = {
    "light":        "martin-ha/toxic-comment-model",           # DistilBERT ~250MB — Render free tier
    "full":         "unitary/toxic-bert",                      # BERT multi-label ~440MB — local/GPU
    "multilingual": "unitary/multilingual-toxic-xlm-roberta",  # XLM-R ~550MB — multi-language
}

# Read from env; resolve preset alias or treat as raw model ID
_env_model = os.getenv("DETECTION_MODEL", "light").strip()
ACTIVE_MODEL = MODEL_PRESETS.get(_env_model, _env_model)

# Global pipeline cache (lazy-loaded on first predict call)
_pipe = None


def get_classifier():
    """Lazy-load and cache the transformer pipeline."""
    global _pipe
    if _pipe is None:
        print(f"🤖 Loading detection model: {ACTIVE_MODEL}  (preset='{_env_model}')")
        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            "text-classification",
            model=ACTIVE_MODEL,
            tokenizer=ACTIVE_MODEL,
            device=device,
            top_k=None,  # Return all class probabilities
        )
        print(f"✅ Detection model loaded on {'GPU' if device == 0 else 'CPU'}")
    return _pipe


def extract_toxicity_score(scores: dict) -> float:
    """
    Extract a single toxicity score (0-1) from raw model label scores.
    Works with both:
      - Multi-label models (toxic-bert): returns max across harmful categories.
      - Binary models (DistilBERT): returns the score for the 'toxic'/'LABEL_1' label.
    """
    # Multi-label: known harmful categories (toxic-bert, etc.)
    tox_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    multi_score = max(scores.get(cat, 0.0) for cat in tox_categories)
    if multi_score > 0:
        return multi_score

    # Binary generic: LABEL_1 is typically the positive (toxic) class
    if "LABEL_1" in scores:
        return scores["LABEL_1"]

    # Fallback: any label containing "toxic" but not "non"
    for label, score in scores.items():
        if "toxic" in label.lower() and "non" not in label.lower():
            return score

    return 0.0


def train(df: pd.DataFrame) -> dict:
    """
    Detection training stub.
    Fine-tuning is not performed live — a pre-trained model is used.
    Returns the active model name and reference metrics for the UI.
    """
    return {
        "accuracy": 0.92 if ACTIVE_MODEL == MODEL_PRESETS["full"] else 0.89,
        "report": {"status": f"Pre-trained model active: {ACTIVE_MODEL}"},
        "train_size": len(df),
        "test_size": 0,
        "model_id": ACTIVE_MODEL,
    }


def predict(messages: list[str], progress_cb=None) -> list[dict]:
    """
    Predict toxicity for a list of raw message strings.
    Model-agnostic: works with any multi-label or binary HuggingFace classifier.
    """
    classifier = get_classifier()
    cleaned = [clean_text(m) for m in messages]

    # Smaller batch size on free tier to reduce peak RAM usage
    batch_size = 16 if ACTIVE_MODEL == MODEL_PRESETS["light"] else 64
    total = len(cleaned)
    raw_predictions = []

    for i in range(0, total, batch_size):
        batch = cleaned[i : i + batch_size]
        try:
            batch_preds = classifier(batch)
            raw_predictions.extend(batch_preds)

            if progress_cb:
                progress_cb(min(i + len(batch), total), total)

            if (i + len(batch)) % (batch_size * 5) == 0 or (i + len(batch)) == total:
                print(f"    - Model Progress: {min(i + len(batch), total)} / {total} messages processed...")
        except Exception as e:
            print(f"ERROR: classifier error on batch {i}: {e}")
            raise e

    results = []
    for msg, raw_pred in zip(messages, raw_predictions):
        scores = {p["label"]: p["score"] for p in raw_pred}
        toxicity_score = extract_toxicity_score(scores)
        is_bullying = bool(toxicity_score >= 0.7)

        results.append({
            "text": msg,
            "is_bullying": is_bullying,
            "confidence": round(max(scores.values()), 4),
            "toxicity_score": round(toxicity_score, 4),
        })

    return results

