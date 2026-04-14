"""
Full Experimental Evaluation Pipeline
======================================
Loads the HateXplain CSV, runs unitary/toxic-bert for real toxicity scores,
trains LSTM + Random Forest, evaluates on held-out test split, prints
sklearn classification_report for each stage.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

# ─── CONFIG ─────────────────────────────────────────────
CSV_PATH = r"g:/TY SEM II/ML_PROJECT/Datasets/final_hateXplain.csv"
BERT_MODEL = "unitary/toxic-bert"
BERT_BATCH = 256
BULLY_THRESHOLD = 0.7
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 20
LSTM_LR = 0.001
MAX_SEQ_LEN = 50
MSGS_PER_CONV = 5
RF_ESTIMATORS = 100
TEST_SIZE = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1. LOAD & INSPECT DATASET ─────────────────────────
print("=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)
df = pd.read_csv(CSV_PATH)
print(f"Columns : {list(df.columns)}")
print(f"Shape   : {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nLabel distribution:")
print(df["label"].value_counts().to_string())

# Binarize: offensive/hatespeech -> 1, normal -> 0
df["binary_label"] = df["label"].apply(lambda x: 1 if str(x).lower() in ["offensive", "hatespeech"] else 0)
print(f"\nBinary label distribution:")
print(df["binary_label"].value_counts().to_string())

# ─── 2. BERT TOXICITY SCORING ──────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Running/Loading unitary/toxic-bert scores")
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 70)

import os
CACHED_CSV = "df_with_scores.csv"

if os.path.exists(CACHED_CSV):
    print("  Loading cached BERT scores from df_with_scores.csv ...")
    df = pd.read_csv(CACHED_CSV)
    bert_time = 0.0
else:
    device_id = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=BERT_MODEL, tokenizer=BERT_MODEL, device=device_id, top_k=None)
    
    texts = df["comment"].fillna("").astype(str).tolist()
    all_toxicity_scores = []
    
    t0 = time.time()
    for i in range(0, len(texts), BERT_BATCH):
        batch = texts[i:i+BERT_BATCH]
        preds = classifier(batch)
        for pred_list in preds:
            scores = {p["label"]: p["score"] for p in pred_list}
            tox_cats = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            tox_score = max(scores.get(c, 0.0) for c in tox_cats)
            all_toxicity_scores.append(tox_score)
        done = min(i + len(batch), len(texts))
        if done % 2000 < BERT_BATCH or done == len(texts):
            print(f"  BERT progress: {done}/{len(texts)} messages scored...")
    
    bert_time = time.time() - t0
    print(f"  BERT scoring complete in {bert_time:.1f}s")
    
    df["toxicity_score"] = all_toxicity_scores
    df["is_bullying_pred"] = (df["toxicity_score"] >= BULLY_THRESHOLD).astype(int)
    
    # Save to cache
    df.to_csv(CACHED_CSV, index=False)
    print("  Saved cached BERT scores to df_with_scores.csv")

# ─── 3. BERT BINARY DETECTION REPORT ──────────────────
print("\n" + "=" * 70)
print("STEP 3: BERT Binary Detection Evaluation")
print(f"Threshold = {BULLY_THRESHOLD}")
print("=" * 70)

bert_report = classification_report(
    df["binary_label"], df["is_bullying_pred"],
    target_names=["Normal", "Bullying"],
    digits=4, output_dict=True
)
bert_report_str = classification_report(
    df["binary_label"], df["is_bullying_pred"],
    target_names=["Normal", "Bullying"],
    digits=4
)
print(bert_report_str)
bert_acc = accuracy_score(df["binary_label"], df["is_bullying_pred"])
print(f"Overall Accuracy: {bert_acc:.4f}")

# ─── 4. BUILD CONVERSATIONS & FEATURES ────────────────
print("\n" + "=" * 70)
print("STEP 4: Building Conversation Windows & Extracting Features")
print("=" * 70)

np.random.seed(SEED)
n = len(df)
df["conversation_id"] = [f"conv_{i // MSGS_PER_CONV}" for i in range(n)]
users = [f"user_{i}" for i in range(1, 101)]
df["user_id"] = np.random.choice(users, n)

ABUSIVE_WORDS = {
    "hate", "stupid", "idiot", "loser", "ugly", "fat", "dumb",
    "kill", "die", "worthless", "pathetic", "disgusting", "freak",
    "moron", "jerk", "bastard", "bitch", "cunt", "fuck", "shit",
    "ass", "damn", "hell", "crap", "retard", "psycho",
}

def count_abusive(text):
    if not isinstance(text, str): return 0
    return len(set(text.lower().split()) & ABUSIVE_WORDS)

def get_sentiment(text):
    try: return TextBlob(str(text)).sentiment.polarity
    except: return 0.0

def extract_features(group):
    n = len(group)
    tox = group["toxicity_score"].fillna(0).values
    avg_tox = float(np.mean(tox))
    max_tox = float(np.max(tox))
    tox_trend = float(np.polyfit(range(n), tox, 1)[0]) if n > 1 else 0.0
    sents = group["comment"].apply(get_sentiment).values
    avg_sent = float(np.mean(sents))
    sent_trend = float(np.polyfit(range(n), sents, 1)[0]) if n > 1 else 0.0
    abuse_counts = group["comment"].apply(count_abusive).values
    abuse_freq = float(np.sum(abuse_counts)) / max(n, 1)
    bully_ratio = float(group["is_bullying_pred"].mean())
    repeated = 0
    if "user_id" in group.columns and n > 2:
        vc = group["user_id"].value_counts()
        if vc.iloc[0] / n > 0.5:
            repeated = 1
    return pd.Series({
        "avg_toxicity": avg_tox, "max_toxicity": max_tox, "toxicity_trend": tox_trend,
        "avg_sentiment": avg_sent, "sentiment_trend": sent_trend,
        "abusive_freq": abuse_freq, "bully_ratio": bully_ratio,
        "repeated_target": repeated, "message_count": n,
    })

conv_features = []
conv_tox_sequences = []
conv_labels = []

for conv_id, group in df.groupby("conversation_id"):
    if len(group) < 2:
        continue
    feats = extract_features(group)
    scores_list = group["toxicity_score"].tolist()

    # Derive ground-truth escalation label from human annotations (ground truth), NOT from BERT scores
    # This prevents data leakage and ensures a valid evaluation metric.
    true_labels = group["label"].str.lower().tolist()
    if "hatespeech" in true_labels:
        label = "HIGH"
    elif "offensive" in true_labels:
        label = "MEDIUM"
    else:
        label = "LOW"

    conv_features.append(feats)
    conv_tox_sequences.append(scores_list)
    conv_labels.append(label)

feat_df = pd.DataFrame(conv_features)
feat_df["label"] = conv_labels
print(f"Total conversations: {len(feat_df)}")
print(f"Escalation label distribution:")
print(pd.Series(conv_labels).value_counts().to_string())

# ─── 5. TRAIN RANDOM FOREST ───────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Training & Evaluating Random Forest Escalation Classifier")
print("=" * 70)

feature_cols = ["avg_toxicity", "max_toxicity", "toxicity_trend", "avg_sentiment",
                "sentiment_trend", "abusive_freq", "bully_ratio", "repeated_target", "message_count"]

X_rf = feat_df[feature_cols].fillna(0).values
le = LabelEncoder()
y_rf = le.fit_transform(feat_df["label"].values)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=TEST_SIZE, random_state=SEED)

clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=SEED)
clf.fit(X_train_rf, y_train_rf)
y_pred_rf = clf.predict(X_test_rf)

rf_report = classification_report(y_test_rf, y_pred_rf, target_names=le.classes_, digits=4, output_dict=True)
rf_report_str = classification_report(y_test_rf, y_pred_rf, target_names=le.classes_, digits=4)
print(rf_report_str)
rf_acc = accuracy_score(y_test_rf, y_pred_rf)
print(f"Overall Accuracy: {rf_acc:.4f}")

# Feature importances
importances = clf.feature_importances_
print("\nFeature Importances:")
for fname, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {fname:20s} : {imp:.4f}")

# ─── 6. TRAIN & EVALUATE LSTM ─────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Training & Evaluating LSTM Sequential Model")
print(f"Device: {DEVICE}")
print("=" * 70)

LABEL_TO_IDX = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
IDX_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

class EscalationLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=LSTM_DROPOUT)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def pad_sequence(scores):
    if len(scores) > MAX_SEQ_LEN:
        scores = scores[-MAX_SEQ_LEN:]
    else:
        scores = [0.0] * (MAX_SEQ_LEN - len(scores)) + scores
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

# Prepare data
samples = [(seq, LABEL_TO_IDX[lbl]) for seq, lbl in zip(conv_tox_sequences, conv_labels)]

# Split into train/test
train_samples, test_samples = train_test_split(samples, test_size=TEST_SIZE, random_state=SEED)

model = EscalationLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LSTM_LR)

model.train()
for epoch in range(LSTM_EPOCHS):
    total_loss = 0.0
    for scores, label in train_samples:
        seq = pad_sequence(scores).to(DEVICE)
        target = torch.tensor([label], dtype=torch.long).to(DEVICE)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_samples)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch [{epoch+1}/{LSTM_EPOCHS}] — Loss: {avg_loss:.4f}")

# Evaluate on test set
model.eval()
lstm_true = []
lstm_pred = []
with torch.no_grad():
    for scores, label in test_samples:
        seq = pad_sequence(scores).to(DEVICE)
        out = model(seq)
        pred = torch.argmax(out, dim=1).item()
        lstm_true.append(label)
        lstm_pred.append(pred)

lstm_report_str = classification_report(lstm_true, lstm_pred, target_names=["HIGH", "LOW", "MEDIUM"], digits=4)
lstm_report = classification_report(lstm_true, lstm_pred, target_names=["HIGH", "LOW", "MEDIUM"], digits=4, output_dict=True)
lstm_acc = accuracy_score(lstm_true, lstm_pred)

print("\nLSTM Test Set Classification Report:")
print(lstm_report_str)
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")

# ─── 7. INFERENCE LATENCY ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Inference Latency Benchmark (single message)")
print("=" * 70)

test_msg = "you are such an ugly stupid idiot go kill yourself"

# BERT
t0 = time.time()
for _ in range(10):
    _ = classifier([test_msg])
bert_lat = (time.time() - t0) / 10 * 1000
print(f"  BERT single-message latency: {bert_lat:.1f} ms")

# LSTM
test_seq = pad_sequence([0.9, 0.85, 0.7, 0.6, 0.5]).to(DEVICE)
model.eval()
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(test_seq)
lstm_lat = (time.time() - t0) / 100 * 1000
print(f"  LSTM single-sequence latency: {lstm_lat:.2f} ms")

# RF
test_feats = np.array([[0.7, 0.9, 0.1, -0.3, -0.05, 0.4, 0.6, 1, 5]])
t0 = time.time()
for _ in range(1000):
    _ = clf.predict(test_feats)
rf_lat = (time.time() - t0) / 1000 * 1000
print(f"  RF single-prediction latency: {rf_lat:.2f} ms")

total_lat = bert_lat + lstm_lat + rf_lat
print(f"  Total pipeline latency: {total_lat:.1f} ms")

# ─── 8. SUMMARY ───────────────────────────────────────
summary_str = f"""
======================================================================
FINAL SUMMARY — Copy these numbers into the paper
======================================================================

--- BERT Binary Detection (threshold={BULLY_THRESHOLD}) ---
Accuracy        : {bert_acc:.4f}
Normal  P/R/F1  : {bert_report['Normal']['precision']:.4f} / {bert_report['Normal']['recall']:.4f} / {bert_report['Normal']['f1-score']:.4f}
Bullying P/R/F1 : {bert_report['Bullying']['precision']:.4f} / {bert_report['Bullying']['recall']:.4f} / {bert_report['Bullying']['f1-score']:.4f}
Macro F1        : {bert_report['macro avg']['f1-score']:.4f}
Weighted F1     : {bert_report['weighted avg']['f1-score']:.4f}

--- Random Forest Tri-Class Escalation ---
Accuracy        : {rf_acc:.4f}
LOW      P/R/F1 : {rf_report['LOW']['precision']:.4f} / {rf_report['LOW']['recall']:.4f} / {rf_report['LOW']['f1-score']:.4f}  (support={int(rf_report['LOW']['support'])})
MEDIUM   P/R/F1 : {rf_report['MEDIUM']['precision']:.4f} / {rf_report['MEDIUM']['recall']:.4f} / {rf_report['MEDIUM']['f1-score']:.4f}  (support={int(rf_report['MEDIUM']['support'])})
HIGH     P/R/F1 : {rf_report['HIGH']['precision']:.4f} / {rf_report['HIGH']['recall']:.4f} / {rf_report['HIGH']['f1-score']:.4f}  (support={int(rf_report['HIGH']['support'])})
Macro F1        : {rf_report['macro avg']['f1-score']:.4f}
Weighted F1     : {rf_report['weighted avg']['f1-score']:.4f}

--- LSTM Sequential Escalation ---
Test Accuracy   : {lstm_acc:.4f}
HIGH     P/R/F1 : {lstm_report['HIGH']['precision']:.4f} / {lstm_report['HIGH']['recall']:.4f} / {lstm_report['HIGH']['f1-score']:.4f}  (support={int(lstm_report['HIGH']['support'])})
LOW      P/R/F1 : {lstm_report['LOW']['precision']:.4f} / {lstm_report['LOW']['recall']:.4f} / {lstm_report['LOW']['f1-score']:.4f}  (support={int(lstm_report['LOW']['support'])})
MEDIUM   P/R/F1 : {lstm_report['MEDIUM']['precision']:.4f} / {lstm_report['MEDIUM']['recall']:.4f} / {lstm_report['MEDIUM']['f1-score']:.4f}  (support={int(lstm_report['MEDIUM']['support'])})
Macro F1        : {lstm_report['macro avg']['f1-score']:.4f}

--- Latency ---
BERT   : {bert_lat:.1f} ms
LSTM   : {lstm_lat:.2f} ms
RF     : {rf_lat:.2f} ms
Total  : {total_lat:.1f} ms

✅ EVALUATION COMPLETE
"""

print(summary_str)

with open("evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write(summary_str)

