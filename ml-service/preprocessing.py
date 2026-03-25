"""
Text preprocessing module.
Handles lowercasing, stopword removal, tokenization,
and slang/abusive word normalization.
"""
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Slang / abusive word normalization dictionary
# Add more entries as needed for your domain
# ---------------------------------------------------------------------------
SLANG_MAP = {
    "u": "you", "ur": "your", "r": "are", "omg": "oh my god",
    "wtf": "what the fuck", "lmao": "laughing my ass off",
    "stfu": "shut the fuck up", "idiot": "idiot", "loser": "loser",
    "stupid": "stupid", "dumb": "dumb", "hate": "hate",
    "kill": "kill", "die": "die", "ugly": "ugly", "fat": "fat",
}

# Basic abusive word list used for frequency counting
ABUSIVE_WORDS = {
    "hate", "stupid", "idiot", "loser", "ugly", "fat", "dumb",
    "kill", "die", "worthless", "pathetic", "disgusting", "freak",
    "moron", "jerk", "bastard", "bitch", "cunt", "fuck", "shit",
    "ass", "damn", "hell", "crap", "retard", "psycho",
}

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove mentions (@user) and hashtags
    4. Remove punctuation and digits
    5. Normalize slang
    6. Remove stopwords
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Normalize slang
    tokens = [SLANG_MAP.get(tok, tok) for tok in tokens]

    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def count_abusive_words(text: str) -> int:
    """Count how many abusive words appear in a (raw) text."""
    if not isinstance(text, str):
        return 0
    tokens = set(text.lower().split())
    return len(tokens & ABUSIVE_WORDS)


def tokenize(text: str) -> list[str]:
    """Return cleaned tokens for a piece of text."""
    return clean_text(text).split()
