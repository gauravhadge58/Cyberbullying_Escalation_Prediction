import os
import json
import shutil
from datetime import datetime

# Centralized location for all trained models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
REGISTRY_FILE = os.path.join(MODELS_DIR, "registry.json")

def ensure_dirs():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def get_registry():
    """Load the current model registry or return a default empty structure."""
    ensure_dirs()
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"active_id": None, "models": []}

def save_registry(data):
    """Save the updated registry dict to disk."""
    ensure_dirs()
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=4)

def _migrate_legacy_models(reg):
    """Check if old generic models exist and migrate them to a v0 registry entry."""
    legacy_rf = os.path.join(MODELS_DIR, "escalation_model.joblib")
    legacy_enc = os.path.join(MODELS_DIR, "label_encoder.joblib")
    legacy_lstm = os.path.join(MODELS_DIR, "lstm_escalation.pth")
    
    # If a legacy Random Forest exists and we don't already have an active registry ID
    if os.path.exists(legacy_rf) and not reg.get("active_id"):
        print("Model Registry: Migrating legacy models to 'v0' entry.")
        v0_id = "v0_legacy"
        
        # We rename them to timestamped (v0) versions so they fit the new structure
        new_rf = os.path.join(MODELS_DIR, f"rf_{v0_id}.joblib")
        new_enc = os.path.join(MODELS_DIR, f"enc_{v0_id}.joblib")
        
        shutil.move(legacy_rf, new_rf)
        if os.path.exists(legacy_enc):
            shutil.move(legacy_enc, new_enc)
            
        lstm_filename = None
        if os.path.exists(legacy_lstm):
            new_lstm = os.path.join(MODELS_DIR, f"lstm_{v0_id}.pth")
            shutil.move(legacy_lstm, new_lstm)
            lstm_filename = os.path.basename(new_lstm)
            
        entry = {
            "id": v0_id,
            "trained_at": datetime.now().isoformat(),
            "label": "Legacy Pre-Registry Models",
            "rf_filename": os.path.basename(new_rf),
            "encoder_filename": os.path.basename(new_enc) if os.path.exists(new_enc) else None,
            "lstm_filename": lstm_filename,
            "metrics": {"migrated": True}
        }
        
        reg["models"].append(entry)
        reg["active_id"] = v0_id
        save_registry(reg)

def get_active_model_paths():
    """Return dict of absolute paths to the active model files."""
    reg = get_registry()
    if not reg.get("models"):
        _migrate_legacy_models(reg)
        reg = get_registry()
        
    active_id = reg.get("active_id")
    if not active_id:
        return None
        
    # Find active entry
    entry = next((m for m in reg.get("models", []) if m["id"] == active_id), None)
    if not entry:
        return None
        
    paths = {
        "id": active_id,
        "rf": os.path.join(MODELS_DIR, entry["rf_filename"]) if entry.get("rf_filename") else None,
        "encoder": os.path.join(MODELS_DIR, entry["encoder_filename"]) if entry.get("encoder_filename") else None,
        "lstm": os.path.join(MODELS_DIR, entry["lstm_filename"]) if entry.get("lstm_filename") else None
    }
    
    # Enrich the returned JSON so UI knows what's actually available
    for key in ["rf", "encoder", "lstm"]:
        if paths[key] and not os.path.exists(paths[key]):
             paths[key] = None
             
    return paths

def register_new_model(model_id, rf_filename=None, encoder_filename=None, lstm_filename=None, metrics=None, set_active=True):
    """Adds a newly trained model bundle to the registry."""
    reg = get_registry()
    
    entry = {
        "id": model_id,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"Model Run {model_id}",
        "rf_filename": rf_filename,
        "encoder_filename": encoder_filename,
        "lstm_filename": lstm_filename,
        "metrics": metrics or {}
    }
    
    # Store availability flags directly on the dict for easy UI rendering
    entry["rf_available"] = bool(rf_filename)
    entry["lstm_available"] = bool(lstm_filename)
    if metrics:
        entry["lstm_accuracy"] = metrics.get("lstm_accuracy")
        entry["num_conversations"] = metrics.get("num_conversations")
    
    reg["models"].append(entry)
    if set_active:
        reg["active_id"] = model_id
        
    save_registry(reg)
    print(f"Model Registry: Registered '{model_id}' (Active: {set_active})")
    return entry

def activate_model(model_id):
    """Swap the active model used for inference."""
    reg = get_registry()
    if any(m["id"] == model_id for m in reg["models"]):
        reg["active_id"] = model_id
        save_registry(reg)
        print(f"Model Registry: Activated '{model_id}'")
        return True
    return False

def delete_model(model_id):
    """Remove a model entry and its underlying files."""
    reg = get_registry()
    models = reg.get("models", [])
    entry = next((m for m in models if m["id"] == model_id), None)
    
    if not entry:
        return False
        
    # Remove files
    for key in ["rf_filename", "encoder_filename", "lstm_filename"]:
        if entry.get(key):
            path = os.path.join(MODELS_DIR, entry[key])
            if os.path.exists(path):
                os.remove(path)
                
    # Remove from list
    reg["models"] = [m for m in models if m["id"] != model_id]
    
    # Auto-switch active_id if needed
    if reg["active_id"] == model_id:
        reg["active_id"] = reg["models"][-1]["id"] if len(reg["models"]) > 0 else None
        
    save_registry(reg)
    print(f"Model Registry: Deleted '{model_id}'")
    return True
