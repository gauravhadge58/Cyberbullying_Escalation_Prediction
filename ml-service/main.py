"""
Main FastAPI application entry point.
Wires up all routers and configures middleware.

Pipeline:
  Chat Message
    → Text Preprocessing
    → BERT Toxicity Detection      (routers/predict.py → models/detection.py)
    → BiLSTM Escalation Analysis   (models/escalation.py)
    → Random Forest Risk Prediction (models/escalation.py)
    → LangChain Explanation Layer  (langchain_moderator/ — background task)
    → WebSocket Broadcast          (Node.js backend)
"""
import os
import logging
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import train, predict, conversations, stats, moderation

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Download required NLTK data on startup
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

app = FastAPI(
    title="CERDS — Cyberbullying Escalation Risk Detection System",
    description=(
        "ML service for cyberbullying detection and escalation prediction. "
        "Pipeline: BERT Toxicity → BiLSTM Escalation → Random Forest Risk → "
        "LangChain Moderation & Reasoning Layer."
    ),
    version="2.0.0",
)

# Allow all origins in development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────
app.include_router(train.router, tags=["Training"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(conversations.router, tags=["Conversations"])
app.include_router(stats.router, tags=["Stats"])
app.include_router(moderation.router, tags=["Moderation"])


# ── Startup: warm up the LangChain moderator lazily in background ──────────
@app.on_event("startup")
async def _startup():
    langchain_enabled = os.getenv("LANGCHAIN_ENABLED", "true").lower() != "false"
    if langchain_enabled:
        logger.info("CERDS ML Service v2.0 starting — LangChain layer ENABLED.")
        logger.info(
            "LLM provider: %s | Model: %s",
            os.getenv("LLM_PROVIDER", "groq"),
            os.getenv("LLM_MODEL", "llama3-8b-8192"),
        )
    else:
        logger.info("CERDS ML Service v2.0 starting — LangChain layer DISABLED (LANGCHAIN_ENABLED=false).")


@app.get("/")
def root():
    return {
        "message": "CERDS ML Service is running",
        "version": "2.0.0",
        "pipeline": "BERT → BiLSTM → RandomForest → LangChain",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
