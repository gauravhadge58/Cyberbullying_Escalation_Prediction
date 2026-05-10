"""
LangChainModerator — core singleton that orchestrates the LLM-powered
moderation reasoning layer for the CERDS pipeline.

Design:
  - Lazy-initialised: the LLM client is created only on first use, so it
    does not block FastAPI startup or the existing ML pipeline.
  - Provider-agnostic: resolves LLM_PROVIDER env var to Groq / OpenAI / Ollama.
  - Async-friendly: `generate_explanation_async` wraps the synchronous call
    in a thread executor so it can be awaited from FastAPI background tasks.
  - Graceful degradation: if the LLM is unreachable, returns a structured
    fallback dict so the frontend never receives an empty moderation field.
"""
import json
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .memory_store import ConversationMemoryStore
from .prompt_templates import MODERATION_PROMPT, SUMMARY_PROMPT
from .knowledge_base import retrieve_policy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────
_memory_store = ConversationMemoryStore()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="langchain-moderator")

# Action labels for mapping
ACTION_LABELS = {
    "monitor": "👁️ Monitor",
    "warn_user": "⚠️ Warn User",
    "temporary_mute": "🔇 Temporary Mute",
    "escalate_to_human": "🚨 Escalate to Human",
}

VALID_ACTIONS = set(ACTION_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm():
    """
    Instantiate the configured LLM provider.
    Reads LLM_PROVIDER, GROQ_API_KEY / OPENAI_API_KEY / OLLAMA_BASE_URL from env.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to ml-service/.env to enable LangChain moderation."
            )
        logger.info("LangChainModerator: Using Groq provider (model=%s).", model)
        return ChatGroq(
            api_key=api_key, 
            model_name=model, 
            temperature=0.3, 
            max_tokens=512,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        openai_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        logger.info("LangChainModerator: Using OpenAI provider (model=%s).", openai_model)
        return ChatOpenAI(api_key=api_key, model_name=openai_model, temperature=0.3, max_tokens=512)

    elif provider == "ollama":
        from langchain_community.llms import Ollama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("LLM_MODEL", "llama3")
        logger.info("LangChainModerator: Using Ollama provider (model=%s, url=%s).", ollama_model, base_url)
        return Ollama(base_url=base_url, model=ollama_model, temperature=0.3)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use: groq | openai | ollama")


# ─────────────────────────────────────────────────────────────────────────────
# LangChainModerator
# ─────────────────────────────────────────────────────────────────────────────

class LangChainModerator:
    """
    Singleton moderator that:
      1. Builds conversation context from memory
      2. Retrieves relevant cyberbullying policy via RAG
      3. Calls the LLM with a structured prompt
      4. Parses and validates the JSON response
      5. Updates memory with the action taken
    """

    _instance: Optional["LangChainModerator"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._llm = None
            cls._instance._chain = None
            cls._instance._summary_chain = None
            cls._instance._enabled = os.getenv("LANGCHAIN_ENABLED", "true").lower() != "false"
        return cls._instance


    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_chain(self):
        """Lazy-initialise the LLM chain on first call."""
        if self._chain is not None:
            return True
        if not self._enabled:
            logger.info("LangChainModerator: Disabled via LANGCHAIN_ENABLED=false.")
            return False
        try:
            self._llm = _build_llm()
            self._chain = MODERATION_PROMPT | self._llm
            self._summary_chain = SUMMARY_PROMPT | self._llm
            logger.info("LangChainModerator: Chain initialised successfully.")
            return True
        except Exception as exc:
            logger.error("LangChainModerator: Failed to initialise chain — %s", exc)
            return False

    def is_healthy(self) -> dict:
        """Return health status dict for /moderation/health endpoint."""
        ok = self._ensure_chain()
        provider = os.getenv("LLM_PROVIDER", "groq")
        return {
            "status": "ok" if ok else "degraded",
            "enabled": self._enabled,
            "provider": provider,
            "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            "chain_ready": self._chain is not None,
        }

    # ------------------------------------------------------------------
    # Core: Explanation Generation
    # ------------------------------------------------------------------

    def generate_explanation(
        self,
        conv_id: str,
        features: dict,
        messages: list,
        escalation_result: dict,
    ) -> dict:
        """
        Generate a structured moderation explanation synchronously.

        Args:
            conv_id:           Conversation identifier
            features:          Feature dict from escalation.predict_conversation()
            messages:          List of raw message dicts (with 'message' key)
            escalation_result: Output of escalation.predict_conversation()

        Returns:
            Dict with keys: explanation, reasoning, suggested_action, summary,
                            action_label, conv_id, provider
        """
        if not self._ensure_chain():
            return self._fallback_response(conv_id, escalation_result)

        # Update memory with new data
        level = escalation_result.get("escalation_level", "LOW")
        _memory_store.update(conv_id, level, features, messages)

        # Build prompt inputs
        recent_texts = [m.get("message", m.get("text", "")) for m in messages[-6:] if m.get("message") or m.get("text")]
        recent_messages_str = "\n".join(f"  - {t}" for t in recent_texts) or "(no messages)"

        memory_context = _memory_store.build_context_string(conv_id)

        # RAG: retrieve relevant policy
        query = f"{level} risk conversation with toxicity trend {features.get('toxicity_trend', 0):.2f} and abusive frequency {features.get('abusive_freq', 0):.2f}"
        policy_context = retrieve_policy(query, k=2)

        prompt_inputs = {
            "risk_level": level,
            "toxicity_trend": f"{features.get('toxicity_trend', 0):.3f}",
            "sentiment_trend": f"{features.get('sentiment_trend', 0):.3f}",
            "abusive_word_frequency": f"{features.get('abusive_freq', 0):.2f}",
            "bully_ratio": f"{features.get('bully_ratio', 0):.2%}",
            "max_toxicity": f"{features.get('max_toxicity', 0):.3f}",
            "repeated_target": "Yes" if features.get("repeated_target", 0) else "No",
            "recent_messages": recent_messages_str,
            "memory_context": memory_context,
            "policy_context": policy_context,
        }

        try:
            raw = self._chain.invoke(prompt_inputs)
            content = raw.content if hasattr(raw, "content") else str(raw)
            result = self._parse_llm_output(content, conv_id, escalation_result)
            # Update memory with the action taken
            _memory_store.set_last_action(conv_id, result["suggested_action"])
            return result
        except Exception as exc:
            logger.error("LangChainModerator: LLM call failed for conv %s — %s", conv_id, exc)
            return self._fallback_response(conv_id, escalation_result)

    async def generate_explanation_async(
        self,
        conv_id: str,
        features: dict,
        messages: list,
        escalation_result: dict,
    ) -> dict:
        """Async wrapper — runs in thread executor to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self.generate_explanation,
            conv_id,
            features,
            messages,
            escalation_result,
        )

    # ------------------------------------------------------------------
    # Conversation Summary
    # ------------------------------------------------------------------

    def summarize_conversation(self, conv_id: str, messages: list, features: dict, level: str) -> str:
        """Generate a 2–3 sentence conversation summary for the monitor dashboard."""
        if not self._ensure_chain():
            return f"Conversation {conv_id} — {level} risk with {len(messages)} messages."

        recent_texts = [m.get("message", m.get("text", "")) for m in messages[-6:]]
        memory_context = _memory_store.build_context_string(conv_id)

        try:
            raw = self._summary_chain.invoke({
                "risk_level": level,
                "message_count": len(messages),
                "bully_ratio": f"{features.get('bully_ratio', 0):.2%}",
                "max_toxicity": f"{features.get('max_toxicity', 0):.3f}",
                "memory_context": memory_context,
                "recent_messages": "\n".join(f"  - {t}" for t in recent_texts) or "(none)",
            })
            content = raw.content if hasattr(raw, "content") else str(raw)
            return content.strip()
        except Exception as exc:
            logger.error("LangChainModerator: Summary failed for conv %s — %s", conv_id, exc)
            return f"Conversation {conv_id} — {level} risk with {len(messages)} messages."

    async def summarize_conversation_async(self, conv_id: str, messages: list, features: dict, level: str) -> str:
        """Async wrapper for summarize_conversation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self.summarize_conversation,
            conv_id, messages, features, level,
        )

    # ------------------------------------------------------------------
    # Memory access (for external clearing)
    # ------------------------------------------------------------------

    @property
    def memory(self) -> ConversationMemoryStore:
        return _memory_store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_llm_output(self, content: str, conv_id: str, escalation_result: dict = None) -> dict:
        """Parse and validate the LLM JSON output, with robust fallback extraction."""
        if escalation_result is None:
            escalation_result = {}

        content = content.strip()

        # Robust extraction of JSON from markdown blocks
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # Fallback: find the first { and last }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]

        try:
            data = json.loads(content)
        except Exception as exc:
            logger.warning("LangChainModerator: Could not parse LLM output for %s. Error: %s", conv_id, exc)
            return self._fallback_response(conv_id, escalation_result)

        # Validate & normalise action
        action = data.get("suggested_action", "monitor").lower().strip()
        if action not in VALID_ACTIONS:
            action = "monitor"

        reasoning_val = data.get("reasoning", "• No reasoning available.")
        if isinstance(reasoning_val, list):
            reasoning_val = "\n".join(str(r) for r in reasoning_val)
        else:
            reasoning_val = str(reasoning_val)

        return {
            "conv_id": conv_id,
            "explanation": str(data.get("explanation", "No explanation generated.")),
            "reasoning": reasoning_val,
            "suggested_action": action,
            "action_label": ACTION_LABELS.get(action, action),
            "summary": str(data.get("summary", "Conversation analysed.")),
            "provider": os.getenv("LLM_PROVIDER", "groq"),
        }

    @staticmethod
    def _fallback_response(conv_id: str, escalation_result: dict) -> dict:
        """Return a safe structured response when the LLM is unavailable."""
        level = escalation_result.get("escalation_level", "UNKNOWN")
        action_map = {"HIGH": "escalate_to_human", "MEDIUM": "warn_user", "LOW": "monitor"}
        action = action_map.get(level, "monitor")
        return {
            "conv_id": conv_id,
            "explanation": (
                f"Automated analysis indicates a {level} risk level for this conversation. "
                "The AI explanation layer is currently unavailable — please review the ML signals manually."
            ),
            "reasoning": (
                "• ML pipeline completed successfully (BERT + BiLSTM + Random Forest).\n"
                "• LangChain explanation layer is offline or not configured.\n"
                "• Please add GROQ_API_KEY to ml-service/.env to enable AI explanations."
            ),
            "suggested_action": action,
            "action_label": ACTION_LABELS.get(action, action),
            "summary": f"{level} risk conversation detected by ML pipeline.",
            "provider": "fallback",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level accessor
# ─────────────────────────────────────────────────────────────────────────────

def get_moderator() -> LangChainModerator:
    """Return the global LangChainModerator singleton."""
    return LangChainModerator()
