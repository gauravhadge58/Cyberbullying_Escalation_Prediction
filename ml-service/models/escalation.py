"""
Escalation prediction model — Hybrid (Rule-Based + Random Forest + LSTM).

Pipeline:
  1. Extract 9 hand-crafted features from the conversation.
  2. Rule-based weighted scoring → LOW / MEDIUM / HIGH.
  3. (Optional) Random Forest trained on those features.
  4. (NEW) LSTM on raw toxicity score sequence for temporal pattern detection.
  5. Hybrid decision: combine all three signals.
"""
import os
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

from preprocessing import count_abusive_words
from models import lstm_model as lstm
import model_registry as registry

# Escalation levels
LEVEL_LOW = "LOW"
LEVEL_MEDIUM = "MEDIUM"
LEVEL_HIGH = "HIGH"

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
# Legacy fallback paths
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "escalation_model.joblib")
ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, "escalation_encoder.joblib")


def get_sentiment(text: str) -> float:
    """Return TextBlob polarity score: -1 (negative) to +1 (positive)."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def extract_conversation_features(messages_df: pd.DataFrame) -> pd.Series:
    """
    Extract escalation features for ONE conversation.
    messages_df must be sorted by timestamp ascending.
    
    Features returned:
    - avg_toxicity: mean toxicity_score across messages
    - max_toxicity: peak toxicity
    - toxicity_trend: slope of toxicity over time (positive = escalating)
    - avg_sentiment: mean sentiment polarity
    - sentiment_trend: slope of sentiment (negative slope = more negative over time)
    - abusive_freq: total abusive words / total messages
    - bully_ratio: fraction of messages labelled bullying
    - repeated_target: 1 if same user_id targeted repeatedly (>50% of messages)
    - message_count: total number of messages
    """
    n = len(messages_df)
    
    # Toxicity scores (from detection model output, default 0 if column absent)
    tox = messages_df["toxicity_score"].fillna(0).values if "toxicity_score" in messages_df.columns else np.zeros(n)
    
    avg_toxicity = float(np.mean(tox))
    max_toxicity = float(np.max(tox))
    toxicity_trend = float(np.polyfit(range(n), tox, 1)[0]) if n > 1 else 0.0
    
    # Sentiment
    sentiments = messages_df["message"].apply(get_sentiment).values
    avg_sentiment = float(np.mean(sentiments))
    sentiment_trend = float(np.polyfit(range(n), sentiments, 1)[0]) if n > 1 else 0.0
    
    # Abusive word frequency
    abusive_counts = messages_df["message"].apply(count_abusive_words).values
    abusive_freq = float(np.sum(abusive_counts)) / max(n, 1)
    
    # Bully ratio (if is_bullying column present)
    bully_ratio = 0.0
    if "is_bullying" in messages_df.columns:
        bully_ratio = float(messages_df["is_bullying"].astype(int).mean())
    
    # Repeated targeting: check if one user_id appears in >50% of messages as potential victim
    repeated_target = 0
    if "user_id" in messages_df.columns and n > 2:
        vc = messages_df["user_id"].value_counts()
        if vc.iloc[0] / n > 0.5:
            repeated_target = 1
    
    return pd.Series({
        "avg_toxicity": avg_toxicity,
        "max_toxicity": max_toxicity,
        "toxicity_trend": toxicity_trend,
        "avg_sentiment": avg_sentiment,
        "sentiment_trend": sentiment_trend,
        "abusive_freq": abusive_freq,
        "bully_ratio": bully_ratio,
        "repeated_target": repeated_target,
        "message_count": n,
    })


def rule_based_escalation(features: dict) -> str:
    """
    Compute escalation level from a feature dict using rule-based thresholds.
    Returns: 'LOW', 'MEDIUM', or 'HIGH'
    """
    score = 0
    
    if features.get("max_toxicity", 0) > 0.9:
        score += 4
    elif features.get("max_toxicity", 0) > 0.7:
        score += 3
    elif features.get("max_toxicity", 0) > 0.5:
        score += 2
    
    if features.get("toxicity_trend", 0) > 0.1:
        score += 2
    
    if features.get("avg_sentiment", 0) < -0.4:
        score += 1
    
    if features.get("sentiment_trend", 0) < -0.1:
        score += 1
    
    if features.get("abusive_freq", 0) > 0.5:
        score += 3
    elif features.get("abusive_freq", 0) > 0.2:
        score += 1
    
    if features.get("bully_ratio", 0) > 0.5:
        score += 3
    elif features.get("bully_ratio", 0) > 0.3:
        score += 1
    
    if features.get("repeated_target", 0):
        score += 2
    
    # Lower threshold from 8 to 7 for HIGH risk
    if score >= 7:
        return LEVEL_HIGH
    elif score >= 4:
        return LEVEL_MEDIUM
    else:
        return LEVEL_LOW


def train(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest escalation classifier.
    Saves with a timestamped filename and registers in model registry.
    """
    df = df.dropna(subset=["conversation_id", "message"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["conversation_id", "timestamp"])

    feature_rows = []
    for conv_id, group in df.groupby("conversation_id"):
        feats = extract_conversation_features(group)
        feats["conversation_id"] = conv_id
        if "escalation_level" in group.columns and group["escalation_level"].notna().any():
            feats["label"] = group["escalation_level"].mode()[0]
        else:
            feats["label"] = rule_based_escalation(feats.to_dict())
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    feature_cols = ["avg_toxicity", "max_toxicity", "toxicity_trend", "avg_sentiment",
                    "sentiment_trend", "abusive_freq", "bully_ratio", "repeated_target", "message_count"]

    X = feat_df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(feat_df["label"].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # Save with timestamp
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rf_filename = f"escalation_model_{ts}.joblib"
    enc_filename = f"escalation_encoder_{ts}.joblib"
    joblib.dump(clf, os.path.join(SAVED_MODELS_DIR, rf_filename))
    joblib.dump(le, os.path.join(SAVED_MODELS_DIR, enc_filename))

    return {
        "report": report,
        "classes": list(le.classes_),
        "rf_filename": rf_filename,
        "enc_filename": enc_filename,
        "model_id_ts": ts,
    }


def predict_conversation(messages: list[dict]) -> dict:
    """
    Hybrid escalation prediction: Rule-based + Random Forest (active) + LSTM (active).
    Loads models from the active registry entry.
    """
    df = pd.DataFrame(messages)
    df["timestamp"] = pd.to_datetime(
        df.get("timestamp", [pd.Timestamp.now()] * len(df)), errors="coerce"
    )
    df = df.sort_values("timestamp")

    features = extract_conversation_features(df)
    feat_dict = features.to_dict()

    # ── 1. Rule-Based ────────────────────────────────────
    rule_level = rule_based_escalation(feat_dict)

    # ── 2. Random Forest (from active registry entry) ────
    rf_level = None
    active = registry.get_active_model_entry()
    rf_path = registry.full_path(active.get("rf_path")) if active else None
    enc_path = registry.full_path(active.get("encoder_path")) if active else None

    # Fallback to legacy paths
    if not rf_path or not os.path.exists(rf_path):
        rf_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
        enc_path = ENCODER_PATH if os.path.exists(ENCODER_PATH) else None

    if rf_path and enc_path and os.path.exists(rf_path) and os.path.exists(enc_path):
        try:
            clf = joblib.load(rf_path)
            le = joblib.load(enc_path)
            feature_cols = [
                "avg_toxicity", "max_toxicity", "toxicity_trend", "avg_sentiment",
                "sentiment_trend", "abusive_freq", "bully_ratio", "repeated_target", "message_count"
            ]
            X = np.array([[feat_dict.get(c, 0) for c in feature_cols]])
            pred = clf.predict(X)[0]
            rf_level = le.inverse_transform([pred])[0]
        except Exception:
            rf_level = None

    # ── 3. LSTM (from active registry entry) ─────────────
    tox_scores = df["toxicity_score"].fillna(0.0).tolist() \
        if "toxicity_score" in df.columns else [0.0]
    lstm_path = registry.full_path(active.get("lstm_path")) if active else None
    if lstm_path and not os.path.exists(lstm_path):
        lstm_path = None
    lstm_result = lstm.predict_escalation(tox_scores, model_path=lstm_path)
    lstm_level = lstm_result["lstm_label"]

    # ── 4. Hybrid Decision ───────────────────────────────
    signals = [rule_level, lstm_level]
    if rf_level:
        signals.append(rf_level)

    if LEVEL_HIGH in signals:
        final_level = LEVEL_HIGH
    elif signals.count(LEVEL_MEDIUM) >= 2:
        final_level = LEVEL_MEDIUM
    else:
        final_level = rule_level

    score_map = {LEVEL_LOW: 2, LEVEL_MEDIUM: 5, LEVEL_HIGH: 9}

    return {
        "escalation_level": final_level,
        "escalation_score": score_map.get(final_level, 2),
        "features": feat_dict,
        "lstm_result": lstm_result,
        "rule_level": rule_level,
        "rf_level": rf_level,
        "active_model_id": active["id"] if active else None,
    }
