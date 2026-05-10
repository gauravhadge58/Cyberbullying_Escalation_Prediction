"""
Conversation memory store for LangChain moderation context.
Maintains a per-conversation ConversationBufferMemory so the LLM
can reason about escalation patterns across multiple messages.
"""
import threading
import time
from typing import Dict, Optional


class ConversationMemoryStore:
    """
    Thread-safe store of short-term conversation memory.

    Each conversation_id gets its own sliding-window context buffer
    that remembers:
      - recent escalation predictions
      - prior moderation actions
      - detected patterns (abusive frequency, repeated targeting, etc.)

    Entries expire after TTL_SECONDS of inactivity to bound memory usage.
    """

    TTL_SECONDS = 7200  # 2 hours

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, conv_id: str, escalation_level: str, features: dict, messages: list) -> None:
        """
        Upsert conversation context after each prediction cycle.

        Keeps a rolling window of the last 5 escalation events and up to
        10 recent messages (text only) for prompt injection.
        """
        with self._lock:
            entry = self._store.get(conv_id, {
                "escalation_history": [],
                "recent_messages": [],
                "last_action": None,
                "created_at": time.time(),
            })

            # Append new escalation event
            entry["escalation_history"].append({
                "level": escalation_level,
                "features_snapshot": {
                    "avg_toxicity": round(features.get("avg_toxicity", 0), 3),
                    "toxicity_trend": round(features.get("toxicity_trend", 0), 3),
                    "abusive_freq": round(features.get("abusive_freq", 0), 3),
                    "bully_ratio": round(features.get("bully_ratio", 0), 3),
                },
                "ts": time.time(),
            })
            # Keep only last 5 escalation events
            entry["escalation_history"] = entry["escalation_history"][-5:]

            # Keep last 10 message texts
            texts = [m.get("message", m.get("text", "")) for m in messages if m.get("message") or m.get("text")]
            entry["recent_messages"] = texts[-10:]
            entry["last_accessed"] = time.time()

            self._store[conv_id] = entry

    def get(self, conv_id: str) -> Optional[dict]:
        """Return the memory context for a conversation, or None if not found / expired."""
        with self._lock:
            entry = self._store.get(conv_id)
            if entry is None:
                return None
            if time.time() - entry.get("last_accessed", 0) > self.TTL_SECONDS:
                del self._store[conv_id]
                return None
            return entry

    def set_last_action(self, conv_id: str, action: str) -> None:
        """Record the most recent moderation action taken for a conversation."""
        with self._lock:
            if conv_id in self._store:
                self._store[conv_id]["last_action"] = action

    def build_context_string(self, conv_id: str) -> str:
        """
        Render the memory context as a compact string for prompt injection.
        Returns an empty string if no memory exists yet.
        """
        entry = self.get(conv_id)
        if not entry:
            return "No prior conversation history available."

        history = entry.get("escalation_history", [])
        last_action = entry.get("last_action")

        parts = []

        if history:
            levels = [h["level"] for h in history]
            parts.append(f"Escalation history (oldest→newest): {' → '.join(levels)}")

            # Highlight upward trend
            if len(levels) >= 2 and levels[-1] == "HIGH" and levels[-2] != "HIGH":
                parts.append("⚠️ Escalation recently jumped to HIGH.")

        if last_action:
            parts.append(f"Last moderation action applied: {last_action}")

        return " | ".join(parts) if parts else "First moderation check for this conversation."

    def clear(self, conv_id: str) -> None:
        """Remove a conversation's memory (e.g., after room is cleared)."""
        with self._lock:
            self._store.pop(conv_id, None)

    def clear_all(self) -> None:
        """Wipe all conversation memory."""
        with self._lock:
            self._store.clear()

    def purge_expired(self) -> int:
        """Remove expired entries. Returns count of purged entries."""
        now = time.time()
        with self._lock:
            expired = [
                cid for cid, entry in self._store.items()
                if now - entry.get("last_accessed", 0) > self.TTL_SECONDS
            ]
            for cid in expired:
                del self._store[cid]
        return len(expired)
