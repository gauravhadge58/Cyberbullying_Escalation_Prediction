"""
Model Registry
==============
Manages versioned model saves and the concept of an "active" model.

Registry file: saved_models/registry.json
{
  "active_id": "20240101_120000",
  "models": [
    {
      "id": "20240101_120000",
      "trained_at": "2024-01-01T12:00:00",
      "rf_path": "escalation_model_20240101_120000.joblib",
      "encoder_path": "escalation_encoder_20240101_120000.joblib",
      "lstm_path": "lstm_escalation_20240101_120000.pth",
      "rf_available": true,
      "lstm_available": true,
      "num_conversations": 1200,
      "lstm_accuracy": 0.87,
      "lstm_final_loss": 0.31,
      "label": "My custom model"   # optional user label
    },
    ...
  ]
}
"""

import os
import json
from datetime import datetime

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
REGISTRY_PATH = os.path.join(SAVED_MODELS_DIR, "registry.json")

# Legacy paths (models saved before the registry was introduced)
LEGACY_RF_PATH = os.path.join(SAVED_MODELS_DIR, "escalation_model.joblib")
LEGACY_ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, "escalation_encoder.joblib")
LEGACY_LSTM_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_escalation.pth")


def _load_registry() -> dict:
    """Load registry from disk, or create a fresh one (migrating legacy files)."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)

    # First run — migrate any legacy models into the registry
    registry = {"active_id": None, "models": []}

    if os.path.exists(LEGACY_RF_PATH) or os.path.exists(LEGACY_LSTM_PATH):
        legacy_id = "legacy"
        entry = {
            "id": legacy_id,
            "trained_at": "Pre-registry",
            "label": "Legacy Model (pre-registry)",
            "rf_path": os.path.basename(LEGACY_RF_PATH) if os.path.exists(LEGACY_RF_PATH) else None,
            "encoder_path": os.path.basename(LEGACY_ENCODER_PATH) if os.path.exists(LEGACY_ENCODER_PATH) else None,
            "lstm_path": os.path.basename(LEGACY_LSTM_PATH) if os.path.exists(LEGACY_LSTM_PATH) else None,
            "rf_available": os.path.exists(LEGACY_RF_PATH),
            "lstm_available": os.path.exists(LEGACY_LSTM_PATH),
            "num_conversations": None,
            "lstm_accuracy": None,
            "lstm_final_loss": None,
        }
        registry["models"].append(entry)
        registry["active_id"] = legacy_id

    _save_registry(registry)
    return registry


def _save_registry(registry: dict):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def get_registry() -> dict:
    return _load_registry()


def get_active_id() -> str | None:
    return _load_registry().get("active_id")


def get_active_model_entry() -> dict | None:
    reg = _load_registry()
    active_id = reg.get("active_id")
    if not active_id:
        return None
    for m in reg.get("models", []):
        if m["id"] == active_id:
            return m
    return None


def register_new_model(
    model_id: str,
    rf_filename: str | None,
    encoder_filename: str | None,
    lstm_filename: str | None,
    metrics: dict,
    set_active: bool = True,
) -> dict:
    """
    Add a new model entry to the registry.
    metrics: dict with keys like num_conversations, lstm_accuracy, lstm_final_loss
    """
    registry = _load_registry()

    entry = {
        "id": model_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "label": f"Model {model_id}",
        "rf_path": rf_filename,
        "encoder_path": encoder_filename,
        "lstm_path": lstm_filename,
        "rf_available": rf_filename is not None,
        "lstm_available": lstm_filename is not None,
        "num_conversations": metrics.get("num_conversations"),
        "lstm_accuracy": metrics.get("lstm_accuracy"),
        "lstm_final_loss": metrics.get("lstm_final_loss"),
        "rf_classes": metrics.get("rf_classes"),
    }

    registry["models"].append(entry)
    if set_active:
        registry["active_id"] = model_id

    _save_registry(registry)
    return entry


def activate_model(model_id: str) -> bool:
    """Set a model as active. Returns True if found and activated."""
    registry = _load_registry()
    ids = [m["id"] for m in registry.get("models", [])]
    if model_id not in ids:
        return False
    registry["active_id"] = model_id
    _save_registry(registry)
    return True


def delete_model(model_id: str) -> bool:
    """Remove a model from registry (does NOT delete files from disk)."""
    registry = _load_registry()
    before = len(registry["models"])
    registry["models"] = [m for m in registry["models"] if m["id"] != model_id]
    if len(registry["models"]) == before:
        return False
    if registry.get("active_id") == model_id:
        # Fall back to most recent remaining
        registry["active_id"] = registry["models"][-1]["id"] if registry["models"] else None
    _save_registry(registry)
    return True


def full_path(filename: str | None) -> str | None:
    if not filename:
        return None
    return os.path.join(SAVED_MODELS_DIR, filename)
