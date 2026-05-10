# LangChain Moderation & Reasoning Layer

Technical reference for the `langchain_moderator` Python package — the LLM-powered explainability and moderation layer in CERDS v2.

## Overview

This package sits **after** the Random Forest risk classification step and before the WebSocket broadcast. It runs **asynchronously** (as a FastAPI `BackgroundTask`) so it never blocks the real-time prediction response.

```
Random Forest Risk Prediction
        │
        ▼  [HTTP response returned immediately]
LangChainModerator.generate_explanation_async()
        │
        ├── ConversationMemoryStore (per-conversation context)
        ├── retrieve_policy() (RAG — FAISS vector store)
        ├── MODERATION_PROMPT (LangChain PromptTemplate)
        │
        ▼
LLM (Groq / OpenAI / Ollama)
        │
        ▼
Structured JSON { explanation, reasoning, suggested_action, summary }
        │
        ▼
WebSocket broadcast → MODERATION_UPDATE → React Dashboard
```

## Module Structure

```
langchain_moderator/
├── __init__.py          — Package exports
├── memory_store.py      — ConversationMemoryStore (thread-safe, TTL-based)
├── prompt_templates.py  — LangChain PromptTemplates (moderation + summary)
├── knowledge_base.py    — FAISS RAG knowledge base (10 policy documents)
└── moderator.py         — LangChainModerator singleton (core orchestrator)
```

## LLM Prompt Design

### MODERATION_PROMPT
Accepts 10 variables extracted from the ML pipeline:

| Variable | Source |
|---|---|
| `risk_level` | Random Forest prediction (LOW/MEDIUM/HIGH) |
| `toxicity_trend` | Slope of toxicity scores over time |
| `sentiment_trend` | Slope of sentiment polarity over time |
| `abusive_word_frequency` | Abusive words per message |
| `bully_ratio` | Fraction of messages flagged bullying |
| `max_toxicity` | Peak toxicity score in conversation |
| `repeated_target` | Whether same user is repeatedly targeted |
| `recent_messages` | Last 6 message texts |
| `memory_context` | Prior escalation history from ConversationMemoryStore |
| `policy_context` | Top-2 retrieved policy snippets from FAISS |

**Output format:** Strict JSON with keys `explanation`, `reasoning`, `suggested_action`, `summary`.

### SUMMARY_PROMPT
Shorter template for the `/moderation/summary/:id` endpoint — generates a 2–3 sentence summary from stored memory context without re-submitting all messages.

## Conversation Memory

`ConversationMemoryStore` is a thread-safe in-process dict that:
- Stores the last 5 escalation events per conversation (level + feature snapshot)
- Stores last 10 message texts
- Records the most recent moderation action taken
- Expires entries after 2 hours of inactivity
- Provides `build_context_string()` for prompt injection

## RAG Knowledge Base

`knowledge_base.py` uses `sentence-transformers/all-MiniLM-L6-v2` to embed 10 curated cyberbullying policy documents into a FAISS index at startup. The `retrieve_policy(query, k=2)` function retrieves the top-2 most semantically relevant policies for each moderation scenario.

**Graceful fallback:** If FAISS or sentence-transformers is unavailable, the first two policy documents are returned verbatim.

## Configuration

All configuration is via environment variables (see `ml-service/.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_ENABLED` | `true` | Master switch — set `false` to skip LangChain entirely |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `ollama` |
| `LLM_MODEL` | `llama3-8b-8192` | Model name for chosen provider |
| `GROQ_API_KEY` | — | Required if `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Required if `LLM_PROVIDER=ollama` |

## Performance Characteristics

| Aspect | Detail |
|---|---|
| Added latency to HTTP response | **0ms** (runs in background after response) |
| LangChain processing time | ~0.5–2s (Groq), ~1–3s (OpenAI), variable (Ollama) |
| WebSocket delivery of result | Immediately after LLM responds |
| Memory footprint | ~50MB (sentence-transformers model + FAISS index) |
| Thread pool size | 2 workers (configurable) |
