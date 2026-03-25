"""
Main FastAPI application entry point.
Wires up all routers and configures middleware.
"""
import os
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import train, predict, conversations, stats

# Load environment variables
load_dotenv()

# Download required NLTK data on startup
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

app = FastAPI(
    title="Cyberbullying Detection API",
    description="ML service for cyberbullying detection and escalation prediction",
    version="1.0.0",
)

# Allow all origins in development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(train.router, tags=["Training"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(conversations.router, tags=["Conversations"])
app.include_router(stats.router, tags=["Stats"])


@app.get("/")
def root():
    return {"message": "Cyberbullying Detection ML Service is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}
