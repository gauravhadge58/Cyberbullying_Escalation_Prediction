"""
LangChain PromptTemplates for the CERDS AI Moderation Layer.

Two templates are provided:
  1. MODERATION_PROMPT  — full explanation + action recommendation
  2. SUMMARY_PROMPT     — compact conversation summary
"""
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# Main Moderation Explanation Template
# ─────────────────────────────────────────────────────────────────────────────

MODERATION_TEMPLATE = """\
You are CERDS-AI, an expert cyberbullying moderation assistant embedded in a \
real-time chat safety platform. You receive structured data from an ML pipeline \
(BERT toxicity detection → BiLSTM escalation analysis → Random Forest risk \
classification) and your job is to produce a clear, actionable moderation report.

=== CONVERSATION RISK DATA ===
Risk Level         : {risk_level}
Toxicity Trend     : {toxicity_trend}  (positive = escalating, negative = de-escalating)
Sentiment Trend    : {sentiment_trend} (negative = more negative over time)
Abusive Word Freq  : {abusive_word_frequency} (words per message)
Bully Ratio        : {bully_ratio} (fraction of messages flagged as bullying)
Max Toxicity Score : {max_toxicity} (0–1 scale)
Repeated Targeting : {repeated_target}

=== RECENT MESSAGES (newest last) ===
{recent_messages}

=== CONVERSATION MEMORY ===
{memory_context}

=== RELEVANT MODERATION POLICY ===
{policy_context}

=== YOUR TASK ===
Produce a JSON object with EXACTLY these four keys:
1. "explanation"      — A 2–3 sentence plain-English paragraph explaining WHY this \
conversation is at risk. Reference specific signals (e.g. rising toxicity, repeated \
insults, targeting the same user).
2. "reasoning"        — An ARRAY of strings (3–5 items). Each string must start \
with "• " and describe an escalation factor that drove the risk classification.
3. "suggested_action" — EXACTLY one of these strings: \
"monitor" | "warn_user" | "temporary_mute" | "escalate_to_human"
4. "summary"          — One sentence (≤ 20 words) summarising the situation for a \
dashboard headline.

Guidelines:
- Be factual and objective. Do not invent data not present above.
- For LOW risk: prefer "monitor".
- For MEDIUM risk: prefer "warn_user" or "monitor".
- For HIGH risk: prefer "temporary_mute" or "escalate_to_human".
- If repeated targeting is detected, always recommend at least "warn_user".
- Output ONLY valid JSON, nothing else. No markdown fences.

JSON output:"""

MODERATION_PROMPT = PromptTemplate(
    input_variables=[
        "risk_level",
        "toxicity_trend",
        "sentiment_trend",
        "abusive_word_frequency",
        "bully_ratio",
        "max_toxicity",
        "repeated_target",
        "recent_messages",
        "memory_context",
        "policy_context",
    ],
    template=MODERATION_TEMPLATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Conversation Summary Template
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_TEMPLATE = """\
You are a cyberbullying safety analyst. Summarise the following conversation \
data in 2–3 sentences for a moderation team dashboard. Be factual and specific.

Risk Level: {risk_level}
Message Count: {message_count}
Bully Ratio: {bully_ratio}
Max Toxicity: {max_toxicity}
Escalation Pattern: {memory_context}

Recent Messages:
{recent_messages}

Write ONLY the summary paragraph, no headings, no JSON."""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=[
        "risk_level",
        "message_count",
        "bully_ratio",
        "max_toxicity",
        "memory_context",
        "recent_messages",
    ],
    template=SUMMARY_TEMPLATE,
)
