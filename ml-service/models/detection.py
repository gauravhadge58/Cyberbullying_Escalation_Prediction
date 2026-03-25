"""
Cyberbullying detection model.
Uses TF-IDF vectorizer + Logistic Regression classifier.
Supports train/save/load/predict lifecycle.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from preprocessing import clean_text

# Path where the trained model is persisted
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "detection_model.joblib")


def get_pipeline() -> Pipeline:
    """Build a fresh TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="auto",
            class_weight="balanced",
        )),
    ])


def train(df: pd.DataFrame) -> dict:
    """
    Train the detection model on a labelled DataFrame.

    Args:
        df: DataFrame with columns ['message', 'label']
            label: 1 = bullying, 0 = non-bullying

    Returns:
        dict with accuracy and classification report
    """
    # Drop rows where message or label is missing
    df = df.dropna(subset=["message", "label"]).copy()
    df["label"] = df["label"].astype(int)

    # Preprocess messages
    df["clean"] = df["message"].apply(clean_text)

    X = df["clean"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Persist model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {"accuracy": round(acc, 4), "report": report, "train_size": len(X_train), "test_size": len(X_test)}


def load_model() -> Pipeline:
    """Load the trained model from disk. Returns None if not trained yet."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict(messages: list[str]) -> list[dict]:
    """
    Predict bullying for a list of raw message strings.

    Returns:
        List of dicts: {text, is_bullying, confidence, toxicity_score}
    """
    model = load_model()
    if model is None:
        # Return a default result if model not trained
        return [
            {"text": m, "is_bullying": False, "confidence": 0.0, "toxicity_score": 0.0}
            for m in messages
        ]

    cleaned = [clean_text(m) for m in messages]
    probs = model.predict_proba(cleaned)
    preds = model.predict(cleaned)

    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        # Index 1 = bullying class probability (toxicity score)
        bully_prob = float(prob[1]) if prob.shape[0] > 1 else float(prob[0])
        # Use 0.7 threshold for is_bullying to reduce false positives
        is_bullying = bool(bully_prob >= 0.7)
        results.append({
            "text": msg,
            "is_bullying": is_bullying,
            "confidence": round(max(prob), 4),
            "toxicity_score": round(bully_prob, 4),
        })
    return results
