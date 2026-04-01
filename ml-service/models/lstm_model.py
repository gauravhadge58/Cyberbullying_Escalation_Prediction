"""
LSTM-based escalation prediction model.

Learns temporal patterns from sequences of toxicity scores.
Classifies a conversation as LOW (0), MEDIUM (1), or HIGH (2).

Architecture:
    Input  → LSTM(hidden=32) → Linear(32→3) → Softmax → Class
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime

# ─── Paths ──────────────────────────────────────────────
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")

# ─── Device (GPU if available) ──────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
print(f"🖥️  LSTM config -> cuda_available={torch.cuda.is_available()}, device={DEVICE}, gpu_name={gpu_name}")

# ─── Hyperparameters ────────────────────────────────────
MAX_SEQ_LEN = 10
HIDDEN_SIZE = 32
NUM_CLASSES = 3
LABELS = ["LOW", "MEDIUM", "HIGH"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# Global cached model (path → model)
_cache: dict = {}


# ════════════════════════════════════════════════════════
# Model Definition
# ════════════════════════════════════════════════════════

class EscalationLSTM(nn.Module):
    """
    Single-layer LSTM that takes a sequence of toxicity scores
    and predicts escalation level (LOW / MEDIUM / HIGH).
    """

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES):
        super(EscalationLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)        # (batch, seq_len, hidden)
        last = lstm_out[:, -1, :]         # use last time-step output
        last = self.dropout(last)
        return self.fc(last)              # (batch, num_classes)


# ════════════════════════════════════════════════════════
# Sequence Preparation
# ════════════════════════════════════════════════════════

def prepare_sequence(toxicity_scores: list, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """
    Convert a list of toxicity floats → LSTM input tensor.

    Steps:
      1. Truncate to last `max_len` scores (most recent context)
      2. Left-pad with 0.0 if shorter than max_len
      3. Shape: (1, max_len, 1)  →  (batch=1, seq_len, features=1)
    """
    scores = list(toxicity_scores)[-max_len:]       # keep most recent
    while len(scores) < max_len:
        scores.insert(0, 0.0)                        # left-pad

    tensor = torch.tensor(scores, dtype=torch.float32)
    return tensor.unsqueeze(0).unsqueeze(-1)         # (1, max_len, 1)


# ════════════════════════════════════════════════════════
# Load / Save
# ════════════════════════════════════════════════════════

def get_lstm_model(model_path: str | None = None) -> EscalationLSTM:
    """
    Return a cached, eval-mode LSTM on DEVICE (load weights from model_path if given).
    Results are cached per path.
    """
    global _cache
    key = model_path or "__default__"
    if key not in _cache:
        m = EscalationLSTM().to(DEVICE)
        if model_path and os.path.exists(model_path):
            try:
                m.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"✅ LSTM loaded from {model_path} → {DEVICE}")
            except Exception as e:
                print(f"⚠️  Could not load LSTM weights: {e}")
        else:
            print(f"ℹ️  LSTM: no weights at {model_path!r} — using untrained model on {DEVICE}.")
        m.eval()
        _cache[key] = m
    return _cache[key]


def _save_lstm_model(model: EscalationLSTM, filename: str) -> str:
    """Persist model weights to disk. Returns absolute path."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    path = os.path.join(SAVED_MODELS_DIR, filename)
    torch.save(model.state_dict(), path)
    print(f"✅ LSTM model saved to {path}")
    return path


# ════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════

def train(df: pd.DataFrame, model_id: str | None = None, progress_cb=None) -> dict:
    """
    Train the LSTM on conversation-level toxicity sequences.
    Saves with a timestamped filename.

    Returns metrics + saved filename.
    """
    if "conversation_id" not in df.columns or "toxicity_score" not in df.columns:
        return {"error": "Need conversation_id and toxicity_score columns for LSTM training."}

    df = df.copy()
    df["toxicity_score"] = pd.to_numeric(df["toxicity_score"], errors="coerce").fillna(0.0)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["conversation_id", "timestamp"])

    samples = []
    for conv_id, group in df.groupby("conversation_id"):
        scores = group["toxicity_score"].tolist()
        if len(scores) < 2:
            continue
        if "escalation_level" in group.columns and group["escalation_level"].notna().any():
            level = group["escalation_level"].mode()[0]
        else:
            max_tox = max(scores)
            if max_tox >= 0.8:
                level = "HIGH"
            elif max_tox >= 0.5:
                level = "MEDIUM"
            else:
                level = "LOW"
        samples.append((scores, LABEL_TO_IDX.get(level, 0)))

    if not samples:
        return {"error": "No valid conversations found for LSTM training."}

    model = EscalationLSTM().to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    print(f"  LSTM training on {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    for epoch in range(num_epochs):
        total_loss = 0.0
        for scores, label in samples:
            seq = prepare_sequence(scores).to(DEVICE)
            target = torch.tensor([label], dtype=torch.long).to(DEVICE)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(samples)
        if progress_cb:
            progress_cb(epoch + 1, num_epochs)
        if (epoch + 1) % 5 == 0:
            print(f"  LSTM Epoch [{epoch+1}/{num_epochs}] — Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    final_loss = 0.0
    with torch.no_grad():
        for scores, label in samples:
            seq = prepare_sequence(scores).to(DEVICE)
            out = model(seq)
            pred = torch.argmax(out, dim=1).item()
            if pred == label:
                correct += 1
        total_loss = 0.0
        for scores, label in samples:
            seq = prepare_sequence(scores).to(DEVICE)
            target = torch.tensor([label], dtype=torch.long).to(DEVICE)
            total_loss += nn.CrossEntropyLoss()(model(seq), target).item()
        final_loss = round(total_loss / len(samples), 4)

    accuracy = round(correct / len(samples), 4)
    gpu_tag = f" [GPU: {torch.cuda.get_device_name(0)}]" if DEVICE.type == "cuda" else " [CPU]"
    print(f"✅ LSTM done. Accuracy: {accuracy:.2%} on {len(samples)} conversations.{gpu_tag}")

    # Save with timestamp (always save CPU-compatible weights)
    ts = model_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lstm_escalation_{ts}.pth"
    model.cpu()  # move to CPU for saving (portable weights)
    _save_lstm_model(model, filename)
    model.to(DEVICE)  # move back to GPU for cache

    # Update cache
    global _cache
    path = os.path.join(SAVED_MODELS_DIR, filename)
    model.eval()
    _cache[path] = model

    return {
        "num_conversations": len(samples),
        "epochs": num_epochs,
        "final_loss": final_loss,
        "training_accuracy": accuracy,
        "lstm_filename": filename,
        "device": str(DEVICE),
    }


# ════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════

def predict_escalation(toxicity_scores: list, model_path: str | None = None) -> dict:
    """
    Given a list of toxicity scores, return the LSTM's escalation prediction.
    Uses model at model_path if specified, otherwise untrained default.
    Runs on GPU if available.
    """
    model = get_lstm_model(model_path)
    seq = prepare_sequence(toxicity_scores).to(DEVICE)

    with torch.no_grad():
        logits = model(seq)
        probs = torch.softmax(logits, dim=1)[0].cpu()  # move to CPU for numpy
        pred_idx = torch.argmax(probs).item()

    label = LABELS[pred_idx]
    confidence = round(probs[pred_idx].item(), 4)
    prob_dict = {LABELS[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)}

    return {
        "lstm_label": label,
        "lstm_confidence": confidence,
        "lstm_probs": prob_dict,
    }
