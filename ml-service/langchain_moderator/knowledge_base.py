"""
RAG Knowledge Base for cyberbullying moderation policies.

Uses an in-memory FAISS vector store seeded with curated policy guidelines.
No external downloads required at runtime beyond the sentence-transformers
package. The embeddings are computed once at first call and cached.

Usage:
    from langchain_moderator.knowledge_base import retrieve_policy
    policy_text = retrieve_policy("repeated insults and threats", k=2)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Moderation Policy Knowledge Base
# Each entry is a distinct, self-contained cyberbullying policy guideline.
# ─────────────────────────────────────────────────────────────────────────────
POLICY_DOCUMENTS = [
    # 1
    "Cyberbullying Policy — Repeated Harassment: Users who send three or more "
    "hostile or abusive messages targeting the same individual within a 10-minute "
    "window should receive a formal warning. On a second offense, a temporary 10-minute "
    "mute should be applied. Persistent offenders must be escalated to a human moderator.",

    # 2
    "Cyberbullying Policy — Threat Detection: Any message containing explicit threats "
    "of physical harm, self-harm encouragement, or death threats must trigger immediate "
    "escalation to a human moderator regardless of the conversation's prior escalation "
    "level. Temporary account suspension may be warranted.",

    # 3
    "Cyberbullying Policy — Hate Speech & Identity Attacks: Messages targeting a user "
    "based on race, gender, religion, sexual orientation, disability, or nationality are "
    "classified as hate speech. A single confirmed hate speech message warrants a warning; "
    "repeated incidents require escalation to human review.",

    # 4
    "Cyberbullying Policy — Toxicity Trend Escalation: When the toxicity score of a "
    "conversation shows a consistent upward trend over three or more consecutive messages, "
    "the system should automatically increase the risk level by one tier (LOW → MEDIUM, "
    "MEDIUM → HIGH) and notify on-call moderators.",

    # 5
    "Cyberbullying Policy — Automated Mute Thresholds: A user is eligible for automated "
    "temporary mute (10 minutes) when: (a) a single message scores above 0.95 toxicity, "
    "or (b) the conversation reaches HIGH escalation level and the user has at least one "
    "bullying-flagged message in the session.",

    # 6
    "Cyberbullying Policy — Contextual Moderation: Moderation decisions should consider "
    "conversation context. A single borderline message in a generally benign conversation "
    "should receive a lighter action (monitor or warn) compared to the same message in an "
    "already-escalated conversation.",

    # 7
    "Cyberbullying Policy — Group Targeting: When multiple users coordinate to send "
    "abusive messages to the same target, the incident severity is automatically elevated "
    "to HIGH. All participating users' accounts should be flagged for human review.",

    # 8
    "Cyberbullying Policy — Victim Protection: When a user is identified as the repeated "
    "target of abusive messages (>50% of flagged messages directed at them), the system "
    "should prioritise their safety. Options include: muting the aggressor, notifying the "
    "target of support resources, or temporarily locking the conversation.",

    # 9
    "Cyberbullying Policy — Gradual Escalation Response: Moderation actions should follow "
    "a graduated approach — Monitor → Warn → Temporary Mute → Human Escalation. Skipping "
    "steps is permitted only when imminent harm (threats, explicit hate speech) is detected.",

    # 10
    "Cyberbullying Policy — Transparency & Appeals: Users who receive automated moderation "
    "actions must be informed of the reason. They have the right to appeal within 24 hours. "
    "All automated moderation decisions and the ML signals that triggered them should be "
    "logged for human audit trail purposes.",
]

# ─────────────────────────────────────────────────────────────────────────────
# FAISS-backed retriever (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────
_retriever = None
_retriever_lock = None


def _get_retriever():
    """Build and cache the FAISS retriever (thread-safe, lazy init)."""
    global _retriever, _retriever_lock
    import threading
    if _retriever_lock is None:
        _retriever_lock = threading.Lock()

    with _retriever_lock:
        if _retriever is not None:
            return _retriever
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_core.documents import Document

            logger.info("KnowledgeBase: Building FAISS index from policy documents…")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            docs = [
                Document(page_content=text, metadata={"index": i})
                for i, text in enumerate(POLICY_DOCUMENTS)
            ]
            vectorstore = FAISS.from_documents(docs, embeddings)
            _retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            logger.info("KnowledgeBase: FAISS index ready (%d documents).", len(POLICY_DOCUMENTS))
        except Exception as exc:
            logger.warning("KnowledgeBase: Could not build FAISS index (%s). RAG disabled.", exc)
            _retriever = None
    return _retriever


def retrieve_policy(query: str, k: int = 2) -> str:
    """
    Retrieve the top-k most relevant policy guidelines for a given moderation query.

    Falls back to returning the first two hard-coded policies if FAISS is unavailable.

    Args:
        query: Free-text description of the situation (e.g. "repeated threats and hate speech")
        k:     Number of policy snippets to return

    Returns:
        A single newline-separated string of policy excerpts.
    """
    retriever = _get_retriever()
    if retriever is None:
        # Graceful fallback — return first two policies verbatim
        logger.debug("KnowledgeBase: Using fallback policies (FAISS unavailable).")
        return "\n\n".join(POLICY_DOCUMENTS[:k])

    try:
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs[:k])
    except Exception as exc:
        logger.warning("KnowledgeBase: Retrieval failed (%s). Using fallback.", exc)
        return "\n\n".join(POLICY_DOCUMENTS[:k])
