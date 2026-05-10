# Cyberbullying Detection & Escalation Prediction 🛡️

A production-ready, full-stack application that detects cyberbullying in real-time and predicts conversation escalation risk. Built with a React dashboard, Node.js WebSocket backend, Python FastAPI ML service, and an LLM-powered moderation reasoning layer via LangChain.

## 🧱 Architecture

### Full Pipeline
```
Chat Message
  → Text Preprocessing (NLTK + Slang Normalization)
  → BERT Toxicity Detection (unitary/toxic-bert)
  → BiLSTM Escalation Sequence Analysis
  → Random Forest Risk Classification (LOW / MEDIUM / HIGH)
  → LangChain Moderation & Reasoning Layer ◄── RAG Policy Knowledge Base
  → WebSocket Broadcast → React Dashboard
```

### Services
1. **Frontend (/frontend):** React, Vite, Tailwind CSS, Chart.js. Provides a live dashboard with real-time AI moderation explanations, chat simulator, analytics, and training UI.
2. **Backend (/backend):** Node.js, Express, MongoDB, WebSockets. Acts as middleware, persisting data to MongoDB and broadcasting real-time predictions and moderation updates.
3. **ML Service (/ml-service):** Python, FastAPI, HuggingFace Transformers, PyTorch, Scikit-learn, LangChain. Handles the full ML pipeline plus the LLM-powered explainability layer.

---

## 🚀 Getting Started (Docker Compose — Recommended)

```bash
docker-compose up --build
```

Access:
- **Frontend UI:** `http://localhost:5173`
- **Backend API:** `http://localhost:5000`
- **ML Service Docs:** `http://localhost:8000/docs`

---

## 💻 Local Setup (Without Docker)

### Prerequisites
- Node.js (v18+)
- Python (3.10+)
- MongoDB running on `localhost:27017`
- Groq API key (free at https://console.groq.com)

### 1. ML Service
```bash
cd ml-service
cp .env.example .env
# Edit .env and set GROQ_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Backend
```bash
cd backend
npm install
npm run dev
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🤖 LangChain AI Moderation Layer

The LangChain integration adds an **Explainable AI** layer on top of the existing ML pipeline. After the Random Forest produces a risk classification, a LangChain chain runs asynchronously to generate:

- **Natural-language explanation** of why the conversation is dangerous
- **Escalation reasoning** bullets (toxicity trend, abusive frequency, repeated targeting, etc.)
- **Suggested moderation action**: `monitor` | `warn_user` | `temporary_mute` | `escalate_to_human`
- **Dashboard summary** headline

### LLM Provider Configuration

| Provider | Speed | Cost | Setup |
|---|---|---|---|
| **Groq** (default) | ~0.5s | **FREE** | Get key at [console.groq.com](https://console.groq.com) |
| OpenAI | ~1–2s | Paid | `OPENAI_API_KEY` |
| Ollama | Varies | Free (local) | Install Ollama locally |

Set in `ml-service/.env`:
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
LLM_MODEL=llama3-8b-8192
```

### Conversation Memory
Each conversation maintains a short-term `ConversationBufferMemory` tracking escalation history and prior moderation actions. This allows the LLM to reason about patterns across multiple messages.

### RAG Policy Knowledge Base
A FAISS vector store is seeded with 10 cyberbullying moderation policy guidelines. The most relevant policies are retrieved and injected into each prompt, making recommendations **policy-aware**.

### New API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/moderation/health` | LLM provider health check |
| `POST` | `/api/moderation/explain` | On-demand AI explanation |
| `GET` | `/api/moderation/summary/:convId` | Conversation summary |
| `POST` | `/api/ml-events` | Internal WS broadcast relay |

---

## 🧠 Training the Models

1. Go to **Train Data** tab (`http://localhost:5173/upload`).
2. Upload the sample dataset at `ml-service/data/sample_dataset.csv`.
3. Click **Start Training**.
4. Models are saved to `ml-service/saved_models/` and immediately active.

---

## 🚨 Features

- **Toxicity Prediction:** BERT-based per-message toxicity scoring.
- **Contextual Escalation Risk:** BiLSTM + Random Forest for conversation-level risk (LOW / MEDIUM / HIGH).
- **LangChain AI Moderation:** Explainable AI reasoning with policy-aware recommendations.
- **Conversation Memory:** Short-term context memory for improved contextual reasoning.
- **RAG Policy KB:** FAISS-based retrieval of relevant cyberbullying moderation policies.
- **Real-Time WebSockets:** Predictions and AI explanations appear instantly on the dashboard.
- **Analytics Dashboard:** Toxicity trends and risk distribution via Chart.js.
