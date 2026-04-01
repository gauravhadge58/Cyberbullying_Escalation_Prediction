"""
LSTM sequence model for temporal escalation pattern recognition.
Trains on sequences of toxicity scores to predict final outcome.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import model_registry as registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  LSTM config -> cuda_available={torch.cuda.is_available()}, device={DEVICE}, gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
MAX_SEQ_LEN = 50
LABEL_TO_IDX = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
IDX_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

class EscalationLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(EscalationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def prepare_sequence(scores):
    # Pad to MAX_SEQ_LEN or truncate
    if len(scores) > MAX_SEQ_LEN:
        scores = scores[-MAX_SEQ_LEN:]
    else:
        scores = [0.0] * (MAX_SEQ_LEN - len(scores)) + scores
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


_cached_model = None
_cached_path = None

def get_lstm_model():
    global _cached_model, _cached_path
    
    paths = registry.get_active_model_paths()
    if not paths or not paths.get("lstm"):
        return None
        
    path = paths["lstm"]
    if _cached_model and _cached_path == path:
        return _cached_model
        
    if not os.path.exists(path):
        return None
        
    print(f"Loading LSTM model from {path} onto {DEVICE}")
    model = EscalationLSTM().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    _cached_model = model
    _cached_path = path 
    return model


def train(df: pd.DataFrame, model_id: str | None = None, progress_cb=None) -> dict:
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
            
            target = torch.tensor([label], dtype=torch.long).to(DEVICE)
            final_loss += criterion(out, target).item()
            
    # save model
    if not model_id:
        import datetime
        model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"lstm_{model_id}.pth"
    path = os.path.join(MODELS_DIR, filename)
    torch.save(model.state_dict(), path)

    return {
        "training_accuracy": correct / len(samples),
        "final_loss": final_loss / len(samples),
        "lstm_filename": filename,
        "num_conversations": len(samples),
        "epochs": num_epochs
    }

def predict_escalation(toxicity_scores: list[float]) -> str:
    model = get_lstm_model()
    if not model:
        return "LOW"
        
    with torch.no_grad():
        seq = prepare_sequence(toxicity_scores).to(DEVICE)
        out = model(seq)
        pred = torch.argmax(out, dim=1).item()
        return IDX_TO_LABEL.get(pred, "LOW")
