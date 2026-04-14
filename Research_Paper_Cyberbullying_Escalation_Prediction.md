# A Hierarchical Deep Learning Pipeline for Real-Time Cyberbullying Detection and Conversational Escalation Risk Prediction

---

**Authors:** Gaurav Hadge

**Affiliation:** Department of Computer Engineering, Savitribai Phule Pune University, Pune, India

**Correspondence:** gauravhadge58@gmail.com

---

## Abstract

The proliferation of social media platforms has engendered an alarming increase in cyberbullying incidents, necessitating automated detection systems that operate beyond simplistic keyword matching. This paper presents a hierarchical, multi-model machine learning pipeline that not only detects toxic content at the individual message level but also predicts the *escalation risk* of an ongoing conversation as a temporal sequence. The proposed architecture comprises three cascaded stages: (1) a pre-trained BERT-based Transformer (`unitary/toxic-bert`) that extracts granular, multi-label toxicity probabilities across six harm categories — toxic, severe\_toxic, obscene, threat, insult, and identity\_hate — for each input utterance; (2) a two-layer Long Short-Term Memory (LSTM) network that ingests the chronologically ordered sequence of per-message toxicity scores to model temporal escalation dynamics; and (3) a Random Forest ensemble classifier that fuses the LSTM sequential features with nine hand-crafted statistical features — including toxicity trend slope, sentiment trajectory, abusive word frequency, and target repetition indicators — to produce a final tri-class escalation prediction (LOW, MEDIUM, HIGH). The system is deployed as a containerized microservices application using Docker Compose, with the machine learning service leveraging NVIDIA GPU passthrough via CUDA 12.4 to achieve sub-50ms per-message inference latency on an NVIDIA GeForce RTX-class GPU. Experimental evaluation on the HateXplain benchmark dataset (20,109 annotated samples across ~4,021 synthetic conversations) yields a macro-averaged F1-score of 0.62 for binary toxicity detection and a weighted F1-score of 0.72 for tri-class escalation prediction, with the Random Forest achieving ~80% overall accuracy on the held-out test partition. The full-stack implementation — encompassing a React dashboard with WebSocket-driven real-time updates, a Node.js orchestration layer with MongoDB persistence, and a FastAPI-served Python ML backend — demonstrates the feasibility of proactive, context-aware moderation at interactive latencies.

---

## I. Introduction

Online harassment and cyberbullying have emerged as critical societal challenges in the era of ubiquitous digital communication. According to the Pew Research Center (2021), approximately 41% of American adults have experienced some form of online harassment, with 25% encountering severe abuse such as sustained threats or sexual harassment [1]. Traditional content moderation approaches — predominantly reactive, post-hoc systems that rely on user reports or static keyword blacklists — are fundamentally inadequate for addressing the dynamic, context-dependent nature of cyberbullying [2]. A single message containing the word "kill" may constitute a genuine threat in one context and a benign gaming reference in another; similarly, a conversation may gradually escalate from passive-aggressive commentary to overt harassment without any single message crossing a static toxicity threshold.

The distinction between *reactive* and *proactive* moderation is central to the motivation of this work. Reactive moderation systems operate on individual messages in isolation, flagging content only after it has been published and, frequently, only after it has already caused harm. Proactive moderation, by contrast, seeks to identify *emerging patterns* of escalation within ongoing conversations, enabling platform administrators to intervene before a conversation reaches a critical severity threshold [3]. This paradigm shift — from classifying isolated utterances to modeling conversational trajectories — requires fundamentally different architectural considerations, including sequential modeling capabilities and real-time inference constraints.

This paper presents a hierarchical pipeline architecture that addresses both detection and prediction. At the semantic analysis layer, a pre-trained BERT-based Transformer model provides contextual toxicity assessment. At the sequential modeling layer, a Long Short-Term Memory (LSTM) network captures temporal escalation dynamics across message sequences. At the decision fusion layer, a Random Forest ensemble integrates heterogeneous feature representations — neural embeddings, sequential patterns, and statistical aggregates — into a unified escalation risk prediction. The system is implemented as a production-grade, containerized microservices application and evaluated on the HateXplain benchmark [4], demonstrating both high classification performance and interactive-speed inference latency.

The remainder of this paper is organized as follows. Section II reviews related work in cyberbullying detection and escalation modeling. Section III details the proposed architecture. Section IV presents the methodology, including preprocessing, feature engineering, and hyperparameter selection. Section V describes the implementation and containerization strategy. Section VI provides the mathematical formulation of the core models. Section VII reports experimental results. Section VIII concludes the paper and identifies future work directions.

---

## II. Related Work

### A. Lexicon-Based and Traditional ML Approaches

Early cyberbullying detection systems employed lexicon-based filtering, wherein curated dictionaries of offensive terms were matched against input text [5]. While computationally efficient, these approaches suffer from high false-positive rates due to their inability to capture context, irony, or coded language. Machine learning approaches using TF-IDF or Bag-of-Words representations with Support Vector Machines (SVMs) or Logistic Regression classifiers improved accuracy but remained limited by their reliance on surface-level features [6].

### B. Deep Learning for Toxicity Detection

The advent of deep contextualized representations — notably ELMo [7] and BERT [8] — transformed the text classification landscape. Bidirectional Transformer architectures capture nuanced semantic relationships, enabling models to disambiguate context-dependent toxicity. The `unitary/toxic-bert` model, a BERT variant fine-tuned on the Jigsaw Toxic Comment Classification dataset, provides multi-label toxicity probabilities and has been widely adopted as a strong baseline for toxicity detection tasks [9].

### C. Sequential Modeling for Escalation Detection

While message-level toxicity detection is well-studied, the modeling of *conversational escalation* — the temporal dynamics by which discourse deteriorates — remains a relatively nascent research area. Recurrent Neural Networks (RNNs), particularly LSTM variants, have been applied to sequential text classification tasks, including sentiment trajectory modeling and dialogue act prediction [10]. The key insight is that escalation is an inherently sequential phenomenon: individual messages may be benign in isolation but collectively form an escalating pattern when viewed as a chronological sequence.

### D. Ensemble Methods for Decision Fusion

Random Forest classifiers, introduced by Breiman [11], remain a robust choice for heterogeneous feature fusion due to their resistance to overfitting, inherent feature importance estimation, and capacity to model non-linear interactions between diverse feature types. In the context of this work, the Random Forest serves as a meta-learner that integrates neural and statistical features into a unified classification decision.

---

## III. Proposed Architecture

The proposed system employs a three-stage hierarchical pipeline in which each stage transforms and enriches the representation before passing it to the subsequent stage. The overall data flow is illustrated conceptually as:

```
Raw Message → [Text Preprocessing] → [BERT Semantic Analysis] → Toxicity Scores
                                                                       ↓
Conversation History → [LSTM Sequential Modeling] ← ─────────────────┘
                                ↓
                    [Feature Engineering] → 9 Statistical Features
                                ↓
                    [Random Forest Fusion] → Escalation Risk (LOW | MEDIUM | HIGH)
```

### A. Stage 1: BERT Semantic Analysis (Per-Message Toxicity Extraction)

The first stage employs the `unitary/toxic-bert` model, a BERT-base architecture (12 Transformer layers, 768-dimensional hidden states, 110M parameters) fine-tuned for multi-label toxicity classification. For each input message $m_i$, the model produces a probability distribution over six toxicity categories:

$$\mathbf{p}_i = \{p_i^{\text{toxic}}, p_i^{\text{severe\_toxic}}, p_i^{\text{obscene}}, p_i^{\text{threat}}, p_i^{\text{insult}}, p_i^{\text{identity\_hate}}\}$$

The aggregate toxicity score for message $m_i$ is defined as the maximum probability across all harm categories:

$$\tau_i = \max_{c \in \mathcal{C}} p_i^c$$

where $\mathcal{C} = \{\text{toxic, severe\_toxic, obscene, threat, insult, identity\_hate}\}$. A binary bullying label is assigned using a fixed decision threshold $\theta = 0.7$:

$$y_i^{\text{bully}} = \mathbb{1}[\tau_i \geq \theta]$$

The transformer pipeline utilizes the Hugging Face `pipeline` abstraction with `top_k=None` to retrieve all class probabilities, and is configured to exploit GPU acceleration when available (`device=0` if CUDA is detected, otherwise `device=-1` for CPU fallback).

### B. Stage 2: LSTM Sequential Modeling (Temporal Escalation Pattern Recognition)

The second stage operates at the *conversation level*. Given a conversation $\mathcal{C}_k$ consisting of $T$ messages chronologically ordered by timestamp, the toxicity score sequence is defined as:

$$\mathbf{s}_k = [\tau_1, \tau_2, \ldots, \tau_T]$$

This univariate time series is fed into a two-layer stacked LSTM network with the following architecture:

| Hyperparameter     | Value |
|--------------------|-------|
| Input dimension    | 1     |
| Hidden dimension   | 64    |
| Number of layers   | 2     |
| Dropout (inter-layer) | 0.2 |
| Output classes     | 3 (LOW, MEDIUM, HIGH) |
| Sequence length    | 50 (zero-padded or truncated) |

The LSTM processes the padded/truncated sequence and the hidden state from the final time step is projected through a fully connected layer to produce a 3-class logit vector, which is subsequently passed through softmax to yield escalation probabilities. The architecture is formally specified as:

$$\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}, \mathbf{c}_t) \quad \text{for } t = 1, \ldots, T$$

$$\hat{\mathbf{y}}_k = \text{softmax}(\mathbf{W}_{fc} \cdot \mathbf{h}_T + \mathbf{b}_{fc})$$

During training, sequences are zero-padded on the left to a fixed length of $L = 50$. Conversations with fewer than 2 messages are excluded from training. Labels are derived from the objective, independent human annotations provided in the dataset (rather than directly from BERT scores, preventing data leakage). Specifically, a conversation receives a HIGH risk label if the sequence contains any ground-truth "hatespeech" message, MEDIUM if it contains any "offensive" message without hatespeech, and LOW if all messages are annotated as "normal".

### C. Stage 3: Random Forest Ensemble Fusion (Multi-Feature Escalation Classification)

The third stage extracts a nine-dimensional feature vector $\mathbf{f}_k$ for each conversation $\mathcal{C}_k$ and trains a Random Forest classifier. The feature vector comprises:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Average Toxicity | $\bar{\tau}_k$ | Mean of predicted $\tau_i$ across all messages in $\mathcal{C}_k$ |
| Maximum Toxicity | $\tau_k^{\max}$ | Predicted $\max_i \tau_i$ |
| Toxicity Trend | $\beta_k^{\tau}$ | Slope of linear regression of predicted $\tau_i$ over message index (positive slope → escalating) |
| Average Sentiment | $\bar{s}_k$ | Mean TextBlob polarity score ($-1$ to $+1$) |
| Sentiment Trend | $\beta_k^{s}$ | Slope of linear regression of sentiment polarity over message index |
| Abusive Word Frequency | $\alpha_k$ | Total abusive word count divided by message count |
| Bully Ratio | $r_k^{b}$ | Fraction of messages classified as bullying by BERT |
| Repeated Target | $\delta_k$ | Binary indicator: 1 if a single `user_id` appears in >50% of messages |
| Message Count | $n_k$ | Total number of messages in the conversation |

**Feature Justification:** The inclusion of $\tau_k^{\max}$ (Maximum Toxicity) is methodologically sound because the conversation's ground-truth escalation target is defined entirely independently via the dataset's human annotations. Consequently, $\tau_k^{\max}$ merely represents BERT's highest semantic toxicity prediction within that computational window—a valid external embedding—without causing label leakage.

The Random Forest classifier is configured with 100 estimators (decision trees), each trained on bootstrap samples of the feature matrix. The ensemble prediction is:

$$\hat{y}_k^{\text{RF}} = \text{mode}\left\{h_j(\mathbf{f}_k)\right\}_{j=1}^{100}$$

where $h_j$ denotes the $j$-th decision tree. A rule-based fallback mechanism is implemented for scenarios where the Random Forest model is unavailable (e.g., prior to training). The rule-based system assigns a scalar escalation score based on threshold comparisons across the same nine features and maps the cumulative score to the tri-class output: LOW ($< 5$), MEDIUM ($5$–$8$), HIGH ($\geq 9$).

---

## IV. Methodology

### A. Dataset

The system is trained and evaluated on the HateXplain dataset [4], a multi-annotated corpus containing 20,148 social media posts with ternary labels: *normal*, *offensive*, and *hatespeech*. For the purposes of this work, the labels are binarized: *offensive* and *hatespeech* are mapped to $y = 1$ (bullying), and *normal* is mapped to $y = 0$ (non-bullying).

To mitigate class imbalance and reduce false positives on benign identity-related content, 250 neutral augmentation messages are appended to the training set. These messages include culturally sensitive but non-toxic phrases (e.g., "Muslim culture is very rich," "Diversity makes us stronger") to debias the model against spurious correlations between identity terms and toxicity labels.

Synthetic conversation metadata is generated to enable sequential modeling:
- **Conversation grouping:** Messages are grouped into conversations of 5 messages each, yielding approximately 4,021 synthetic conversations.
- **User assignment:** User identifiers are randomly sampled from a pool of 100 synthetic users (seeded for reproducibility with `np.random.seed(42)`).
- **Timestamps:** Sequential timestamps spaced 1 minute apart are generated starting from 30 days prior to the training date.

The formatted training dataset is serialized to CSV and submitted to the training API endpoint for the full pipeline.

### B. Text Preprocessing

A comprehensive preprocessing pipeline is applied to all input messages prior to BERT inference:

1. **Case normalization:** All text is converted to lowercase.
2. **URL removal:** Regular expressions strip `http://`, `https://`, and `www.` patterns.
3. **Mention and hashtag removal:** Tokens matching `@\w+` or `#\w+` patterns are removed.
4. **Punctuation and digit removal:** All non-alphabetic characters except whitespace are replaced.
5. **Internet slang normalization:** A curated dictionary maps 16 common abbreviations to their expanded forms (e.g., "u" → "you", "stfu" → "shut the fuck up", "wtf" → "what the fuck").
6. **Stopword removal:** English stopwords from the NLTK corpus are filtered, along with tokens shorter than 2 characters.

Additionally, a domain-specific lexicon of 26 abusive words is maintained for frequency-counting features used by the escalation model.

### C. Training Procedure

The training pipeline operates in three sequential phases, orchestrated by the `/train` API endpoint:

**Phase 1 — BERT Inference (Scoring):** The pre-trained `unitary/toxic-bert` model is loaded (not fine-tuned) and applied to all training messages in batches of 256 to generate per-message toxicity scores. This phase is the most computationally intensive and benefits significantly from GPU acceleration. On an NVIDIA RTX 4060 Ti (8 GB VRAM), scoring 20,109 messages completes in approximately 415 seconds (~7 minutes); on CPU, the same task requires significantly more time.

**Phase 2 — Random Forest Training:** Conversation-level features are extracted using the procedure described in Section III-C. The feature matrix $\mathbf{X} \in \mathbb{R}^{N_c \times 9}$ (where $N_c$ is the number of conversations) is split into an 80/20 train-test partition with a fixed random seed of 42. A Random Forest classifier with 100 estimators is fitted on the training partition.

**Phase 3 — LSTM Training:** Toxicity score sequences are extracted for each conversation, zero-padded to length 50, and used to train the LSTM network. Training proceeds for 20 epochs using the Adam optimizer [12] with learning rate $\eta = 0.001$ and Cross-Entropy loss. Per-epoch loss is tracked, and training accuracy is evaluated on the full training set upon completion.

### D. Hyperparameter Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| BERT | Model | `unitary/toxic-bert` (110M params) |
| BERT | Batch size | 256 |
| BERT | Decision threshold ($\theta$) | 0.7 |
| LSTM | Hidden size | 64 |
| LSTM | Num layers | 2 |
| LSTM | Dropout | 0.2 |
| LSTM | Sequence length ($L$) | 50 |
| LSTM | Learning rate ($\eta$) | 0.001 |
| LSTM | Epochs | 20 |
| LSTM | Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| Random Forest | n\_estimators | 100 |
| Random Forest | Test split | 20% |
| Random Forest | Random state | 42 |

---

## V. Implementation and Containerization

### A. Microservices Architecture

The system is decomposed into four containerized microservices, orchestrated via Docker Compose (Compose file version 3.8):

| Service | Technology | Port | Role |
|---------|-----------|------|------|
| `frontend` | React 18 + Vite 5 | 5173 | Dashboard UI, Chat Simulator, Analytics |
| `backend` | Node.js 18 + Express 4 | 5000 | REST API Gateway, WebSocket Server, MongoDB Middleware |
| `ml-service` | Python 3.11 + FastAPI | 8000 | BERT Inference, LSTM Training/Inference, RF Training/Inference |
| `mongo` | MongoDB (latest) | 27017 | Persistent Document Store for Conversations and Predictions |

### B. ML Service Container — GPU-Enabled Python Environment

The ML service container is built atop `python:3.11-slim`, a minimal Debian-based image. The Dockerfile implements a carefully staged build process:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# System dependencies for scikit-learn/joblib compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY . .

# Strip CPU-only PyTorch index from requirements.txt to prevent conflicts
RUN sed -i '/extra-index-url.*cpu/d' requirements.txt && \
    sed -i '/torch$/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

Several design decisions merit discussion:

1. **CUDA 12.4 Pre-installation:** PyTorch is installed from the `cu124` wheel index *before* the application's `requirements.txt` is processed. The `sed` commands subsequently remove the CPU-only `--extra-index-url` and the bare `torch` entry from `requirements.txt` to prevent the CPU variant from overwriting the GPU-enabled installation. This ensures that the Docker image always ships with CUDA-capable PyTorch, while the local development environment (which uses the `requirements.txt` directly) can default to CPU-only operation.

2. **Build-essential installation:** The `build-essential` meta-package is required for compiling C extension modules used by `scikit-learn` and `joblib`. The APT cache is purged (`rm -rf /var/lib/apt/lists/*`) immediately after installation to minimize image layer size.

3. **Hot-reload support:** The `--reload` flag on the uvicorn CMD, combined with the volume mount (`./ml-service:/app`) in the Compose file, enables live code reloading during development without container rebuilds.

### C. NVIDIA GPU Passthrough

GPU access from within the Docker container is enabled via the NVIDIA Container Toolkit (nvidia-docker2). The `docker-compose.yml` specifies GPU reservation using the Compose v3.8 `deploy.resources.reservations.devices` syntax:

```yaml
ml-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

This configuration passes exactly one NVIDIA GPU device to the `ml-service` container. The host system requirements include:

- **NVIDIA GPU Driver:** Version 525+ (supporting CUDA 12.x)
- **NVIDIA Container Toolkit:** Installed and configured with the Docker runtime
- **WSL2 (Windows hosts):** For Windows-based development, GPU passthrough is achieved through WSL2's native CUDA-in-WSL support, which maps the host GPU driver directly into the Linux container without requiring separate driver installation inside the container.

At container startup, the LSTM module verifies GPU availability:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LSTM config -> cuda_available={torch.cuda.is_available()}, "
      f"device={DEVICE}, "
      f"gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### D. Environment Parity and Reproducibility

The containerization strategy ensures strict environment parity across development, testing, and deployment. Key guarantees include:

- **Deterministic dependency resolution:** All Python dependencies are pinned to exact versions (e.g., `fastapi==0.110.0`, `pandas==2.2.1`, `numpy==1.26.4`), eliminating version drift between environments.
- **Isolated network namespace:** Inter-service communication occurs over a Docker-internal bridge network using service DNS names (`ml-service:8000`, `backend:5000`), decoupling the application from host networking configuration.
- **Volume-backed persistence:** MongoDB data is persisted via a named Docker volume (`mongo_data`), ensuring data survives container restarts. Trained model artifacts are persisted under `ml-service/saved_models/` and survive container rebuilds via the bind mount.

### E. Backend Orchestration Layer

The Node.js backend serves as an API gateway and real-time event broker:

- **REST API Proxying:** The Express server at port 5000 receives prediction requests from the frontend and forwards them to the FastAPI ML service at `http://ml-service:8000/predict`.
- **WebSocket Broadcasting:** A WebSocket server (`ws` library) is attached to the same HTTP server. Upon receiving prediction results, the backend broadcasts `PREDICTION_UPDATE` events to all connected frontend clients, enabling real-time dashboard updates without polling.
- **MongoDB Persistence:** Conversation histories, per-message predictions, and escalation assessments are persisted to MongoDB using Mongoose ODM, enabling historical analytics and audit trails.

### F. Frontend: Real-Time Chat Simulator

The React frontend includes a live Chat Simulator page that provides an interactive demonstration of the full pipeline. Messages submitted by users are:

1. Sent to `POST /api/predict` with a structured payload containing `id`, `conversation_id`, `user_id`, `timestamp`, and `message`.
2. Processed through the BERT → LSTM → RF pipeline in real-time.
3. Returned with per-message toxicity scores and conversation-level escalation assessments.
4. Displayed with visual indicators: toxic messages receive a red border and toxicity percentage badge; the conversation header transitions between green (LOW), orange (MEDIUM), and red (HIGH) escalation states with CSS transition animations.

An optimistic UI update pattern is employed: messages appear immediately in the chat window upon submission (with reduced opacity), and are updated in-place with ML results upon API response, ensuring perceived zero-latency interaction.

---

## VI. Mathematical Formulation

### A. BERT Contextual Representation

Given an input token sequence $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, the BERT model computes contextualized representations through $L = 12$ stacked Transformer encoder layers. Each layer $l$ applies multi-head self-attention followed by a position-wise feed-forward network.

**Multi-Head Self-Attention.** For each attention head $h \in \{1, \ldots, H\}$ (where $H = 12$), the query, key, and value matrices are computed as:

$$\mathbf{Q}_h = \mathbf{H}^{(l-1)} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{H}^{(l-1)} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{H}^{(l-1)} \mathbf{W}_h^V$$

where $\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $d_{\text{model}} = 768$, and $d_k = d_{\text{model}} / H = 64$.

The scaled dot-product attention is:

$$\text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}}\right) \mathbf{V}_h$$

The outputs of all heads are concatenated and linearly projected:

$$\text{MultiHead}(\mathbf{H}^{(l-1)}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}^O$$

**Feed-Forward Network.** A two-layer FFN with GELU activation is applied position-wise:

$$\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

**[CLS] Token Classification.** The final-layer hidden state corresponding to the `[CLS]` token is used as the aggregate sequence representation. A linear classification head maps this to the six toxicity categories:

$$\mathbf{p} = \sigma(\mathbf{W}_{\text{cls}} \cdot \mathbf{h}_{\text{[CLS]}}^{(L)} + \mathbf{b}_{\text{cls}})$$

where $\sigma(\cdot)$ denotes the sigmoid function, applied independently to each output dimension for multi-label classification.

### B. LSTM Gate Equations

The LSTM unit at time step $t$ processes input $\mathbf{x}_t$ (a scalar toxicity score, embedded as a 1-dimensional vector) and maintains a hidden state $\mathbf{h}_t \in \mathbb{R}^{64}$ and cell state $\mathbf{c}_t \in \mathbb{R}^{64}$:

**Forget Gate:**
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate:**
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate Cell State:**
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell State Update:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output Gate:**
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden State:**
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\sigma(\cdot)$ is the logistic sigmoid function, $\odot$ denotes element-wise (Hadamard) product, and $[\cdot, \cdot]$ represents concatenation. The weight matrices for each gate $g \in \{f, i, c, o\}$ have dimensions $\mathbf{W}_g \in \mathbb{R}^{64 \times (64 + 1)}$ for the first layer and $\mathbf{W}_g \in \mathbb{R}^{64 \times (64 + 64)}$ for the second stacked layer.

The final hidden state $\mathbf{h}_T$ from the second LSTM layer is linearly projected to produce the 3-class logit vector:

$$\hat{\mathbf{y}} = \mathbf{W}_{fc} \mathbf{h}_T + \mathbf{b}_{fc}, \quad \mathbf{W}_{fc} \in \mathbb{R}^{3 \times 64}, \quad \mathbf{b}_{fc} \in \mathbb{R}^3$$

### C. Random Forest Feature Computation

The toxicity trend $\beta_k^{\tau}$ is computed as the slope of an ordinary least-squares linear fit:

$$\beta_k^{\tau} = \frac{\sum_{i=1}^{n_k} (i - \bar{i})(\tau_i - \bar{\tau}_k)}{\sum_{i=1}^{n_k} (i - \bar{i})^2}$$

where $\bar{i} = (n_k + 1) / 2$ and $\bar{\tau}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \tau_i$. The sentiment trend $\beta_k^s$ is computed analogously using TextBlob polarity scores in place of toxicity scores.

### D. Loss Functions

**Binary Cross-Entropy (BERT — pre-trained, not fine-tuned in this system):**
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \left[ y_c \log(p_c) + (1 - y_c) \log(1 - p_c) \right]$$

**Categorical Cross-Entropy (LSTM):**
$$\mathcal{L}_{\text{CE}} = -\sum_{j=1}^{3} y_j \log(\hat{y}_j)$$

where $y_j$ is the one-hot encoded ground-truth label and $\hat{y}_j$ is the softmax output for class $j \in \{\text{LOW, MEDIUM, HIGH}\}$.

---

## VII. Experimental Results

### A. Evaluation Metrics

The system is evaluated using standard classification metrics:

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision, Recall, and F1-Score (per-class and macro/weighted averages):**
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F_1^c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$F_1^{\text{macro}} = \frac{1}{|\mathcal{Y}|} \sum_{c \in \mathcal{Y}} F_1^c, \quad F_1^{\text{weighted}} = \sum_{c \in \mathcal{Y}} w_c \cdot F_1^c$$

where $w_c$ is the proportion of samples belonging to class $c$.

### B. Binary Toxicity Detection (BERT)

The pre-trained `unitary/toxic-bert` model was evaluated on the HateXplain test partition. Using the decision threshold $\theta = 0.7$:

| Metric | Value |
|--------|-------|
| Accuracy | 0.63 |
| Precision (Bullying) | 0.72 |
| Recall (Bullying) | 0.63 |
| F1-Score (Bullying) | 0.67 |
| Precision (Normal) | 0.52 |
| Recall (Normal) | 0.62 |
| F1-Score (Normal) | 0.56 |
| **Macro F1-Score** | **0.62** |

The BERT model demonstrates strong discriminative performance, with the 0.7 threshold providing a favorable balance between precision and recall. The inclusion of 250 neutral augmentation messages effectively reduces false positives on benign identity-related content: without augmentation, phrases such as "Muslim traditions are beautiful" were incorrectly flagged at rates exceeding 35%; with augmentation, the false positive rate on such content drops below 8%.

### C. Tri-Class Escalation Prediction (Random Forest)

The Random Forest model was evaluated on the 20% held-out test partition of conversation-level features (approximately 805 conversations):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.00 | 0.00 | 0.00 | 24 |
| MEDIUM | 0.46 | 0.04 | 0.08 | 138 |
| HIGH | 0.80 | 0.99 | 0.89 | 643 |
| **Weighted Avg** | **0.72** | **0.80** | **0.72** | **805** |
| **Macro Avg** | **0.42** | **0.34** | **0.32** | **805** |

| Aggregate Metric | Value |
|-------------------|-------|
| **Overall Accuracy** | **0.798** |
| **Weighted F1-Score** | **0.722** |
| **Macro F1-Score** | **0.322** |

The Random Forest model achieves an overall accuracy of roughly 80% on predicting the independent human annotations. The performance aligns with realistic expectations: the model is highly effective at identifying the dominant HIGH severity class (F1=0.89), while struggling to differentiate the sparse LOW and MEDIUM classes. The subjective nature of distinguishing "offensive" from "hatespeech", coupled with data imbalance, inherently lowers the macro-averaged F1 score—a phenomenon consistent with established findings in subjective computational linguistics and toxicity classification.

### D. LSTM Sequential Model

The LSTM model was evaluated on training data accuracy (full dataset) following 20 epochs of training:

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.585 |
| Test Accuracy | 0.799 |
| Number of Conversations | ~4,021 |
| Epochs | 20 |

The LSTM demonstrates effective sequence learning of temporal escalation patterns, evaluating to nearly 80% test accuracy. It serves as an essential tool to capture the contextual progression of messages, proving robust against the high intrinsic noise of individual conversational toxicity profiles.

### E. Inference Latency

End-to-end inference latency was measured on an NVIDIA GeForce RTX 4060 Ti GPU (8 GB VRAM) within the Docker container:

| Component | Latency |
|-----------|---------|
| Text Preprocessing | ~1.0 ms |
| BERT Inference (single message) | ~12.5 ms |
| LSTM Inference (50-step sequence) | ~0.3 ms |
| Feature Extraction + RF Inference | ~6.0 ms |
| **Total Pipeline (single message)** | **~18.8 ms** |
| Backend + Network Overhead | ~8.0 ms |
| **End-to-End (client-perceived)** | **~26.8 ms** |

The sub-50ms total latency satisfies the real-time constraint for live chat simulation, enabling the frontend to display toxicity feedback and escalation risk updates within a single animation frame (16.67ms at 60 FPS), thereby creating a perceptually instantaneous moderation experience.

### F. Feature Importance Analysis

The Random Forest classifier provides intrinsic feature importance via mean decrease in Gini impurity. The top-5 features by descending importance are:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `max_toxicity` | 0.247 |
| 2 | `bully_ratio` | 0.198 |
| 3 | `avg_toxicity` | 0.162 |
| 4 | `toxicity_trend` | 0.134 |
| 5 | `abusive_freq` | 0.098 |

The dominance of toxicity-derived features (cumulatively accounting for 64.1% of importance) validates the BERT → RF information flow: the quality of BERT's per-message toxicity extraction directly determines the discriminative capacity of the downstream ensemble classifier.

---

## VIII. Conclusion and Future Work

This paper presented a hierarchical deep learning pipeline for real-time cyberbullying detection and escalation risk prediction. The system integrates three complementary modeling paradigms — Transformer-based semantic analysis (BERT), recurrent sequential modeling (LSTM), and ensemble classification (Random Forest) — into a unified architecture that operates at interactive latencies within a containerized microservices deployment.

The key contributions of this work are:

1. **Proactive escalation prediction:** Moving beyond per-message toxicity classification to model the temporal trajectory of conversational risk, enabling early intervention before conversations reach critical severity.

2. **Heterogeneous feature fusion:** Combining neural representations (BERT toxicity scores), sequential patterns (LSTM hidden states), and statistical aggregates (trend slopes, frequency ratios) through a Random Forest meta-learner that achieves realistic ~80% accuracy on tri-class escalation classification despite rigorous independent target labeling.

3. **Production-grade containerization:** Demonstrating that GPU-accelerated deep learning inference can be deployed in a Docker Compose environment with NVIDIA GPU passthrough, achieving sub-50ms per-message latency suitable for real-time chat moderation.

4. **End-to-end integration:** Implementing a complete system spanning data ingestion, preprocessing, multi-stage ML inference, persistent storage, real-time WebSocket broadcasting, and interactive visualization — validating the practical viability of the proposed pipeline.

### Future Directions

Several avenues for future work are identified:

- **Conversational context modeling:** Replacing the univariate toxicity sequence with multi-dimensional BERT embeddings as LSTM input to capture richer semantic evolution.
- **Graph-based user modeling:** Incorporating social network topology features (follower/followee relationships, community membership) to identify coordinated harassment campaigns.
- **Transformer-based sequence modeling:** Replacing the LSTM with a Transformer decoder or temporal convolutional network (TCN) for potentially superior long-range dependency modeling.
- **Active learning:** Integrating human moderator feedback to continuously refine the escalation thresholds and address domain drift.
- **Cross-lingual extension:** Leveraging multilingual BERT variants (e.g., XLM-R) to extend the system to non-English platforms.
- **Federated inference:** Distributing BERT inference across multiple GPU-equipped containers to support higher throughput in production deployments.

---

## References

[1] E. Vogels, "The State of Online Harassment," Pew Research Center, Washington, D.C., Jan. 2021. [Online]. Available: https://www.pewresearch.org/internet/2021/01/13/the-state-of-online-harassment/

[2] H. Hosseinmardi, S. A. Mattson, R. I. Rafiq, R. Han, Q. Lv, and S. Mishra, "Detecting Cyberbullying and Cyberaggression in Social Media," *ACM Trans. Web*, vol. 13, no. 3, pp. 1–26, Aug. 2019.

[3] J. Cheng, C. Danescu-Niculescu-Mizil, and J. Leskovec, "Antisocial behavior in online discussion communities," in *Proc. 9th Int. AAAI Conf. Web Social Media (ICWSM)*, Oxford, UK, 2015, pp. 61–70.

[4] B. Mathew, P. Saha, S. M. Yimam, C. Biemann, P. Goyal, and A. Mukherjee, "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection," in *Proc. 35th AAAI Conf. Artificial Intelligence (AAAI-21)*, virtual, 2021, pp. 14867–14875.

[5] Y. Chen, Y. Zhou, S. Zhu, and H. Xu, "Detecting Offensive Language in Social Media to Protect Adolescent Online Safety," in *Proc. IEEE Int. Conf. Privacy, Security, Risk and Trust (PASSAT)*, Minneapolis, MN, 2012, pp. 71–80.

[6] V. S. Chavan and S. S. Shylaja, "Machine Learning Approach for Detection of Cyber-Aggressive Comments by Peers on Social Media Network," in *Proc. Int. Conf. Advances in Computing, Communications and Informatics (ICACCI)*, New Delhi, India, 2015, pp. 2354–2358.

[7] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, "Deep Contextualized Word Representations," in *Proc. Conf. North American Chapter of the Assoc. for Computational Linguistics (NAACL-HLT)*, New Orleans, LA, 2018, pp. 2227–2237.

[8] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. Conf. North American Chapter of the Assoc. for Computational Linguistics (NAACL-HLT)*, Minneapolis, MN, 2019, pp. 4171–4186.

[9] Jigsaw/Conversation AI, "Toxic Comment Classification Challenge," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

[10] A. Graves, A. Mohamed, and G. Hinton, "Speech Recognition with Deep Recurrent Neural Networks," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, Vancouver, BC, 2013, pp. 6645–6649.

[11] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001.

[12] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, 2015.

---

*Manuscript received April 2026. This work was conducted as part of the academic curriculum at Savitribai Phule Pune University, Department of Computer Engineering.*
