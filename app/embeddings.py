"""Text embeddings.

The default embedder is a deterministic hashed bag of tokens. It needs no
network and no model download, which keeps the assignment runnable anywhere.
Set ASAMA_EMBEDDINGS=openai to call text-embedding-3-small at 128 dimensions
before any agents are seeded. Mixing the two providers in one database makes
cosine meaningless, so cosine() returns 0 when widths differ.
"""

from __future__ import annotations

import math
import os
import re
import zlib

EMBED_DIM = 128

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "on",
    "in",
    "with",
    "all",
    "during",
    "this",
    "our",
    "we",
    "you",
    "your",
    "from",
    "by",
    "at",
    "is",
    "are",
    "be",
    "as",
    "that",
    "it",
    "its",
    "into",
    "over",
    "above",
    "their",
    "they",
    "will",
    "can",
    "via",
    "per",
    "when",
    "who",
    "new",
    "across",
    "including",
}


def tokenize(text: str) -> list[str]:
    parts = re.findall(r"[a-z0-9]+", text.lower())
    return [part for part in parts if part not in STOPWORDS and not part.isdigit() and len(part) > 1]


def jaccard(left: str, right: str) -> float:
    a = set(tokenize(left))
    b = set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Signed feature hashing over unigrams and bigrams. Same input, same vector."""
    vector = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        index = digest % dim
        sign = 1.0 if ((digest >> 16) & 1) == 0 else -1.0
        vector[index] += sign
    for first, second in zip(tokens, tokens[1:]):
        digest = zlib.crc32(f"{first}_{second}".encode("utf-8")) & 0xFFFFFFFF
        index = digest % dim
        sign = 1.0 if ((digest >> 8) & 1) == 0 else -1.0
        vector[index] += 0.5 * sign
    return normalize(vector)


def openai_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    import httpx

    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "text-embedding-3-small", "input": text, "dimensions": dim},
        timeout=20.0,
    )
    response.raise_for_status()
    return normalize(list(response.json()["data"][0]["embedding"]))


def embed_text(text: str, concepts: dict[str, float] | None = None) -> list[float]:
    """Embed post or manifesto text. Concept labels are appended so the vector
    agrees with the ontology when the local hasher is in use."""
    labels = " ".join(name.replace("_", " ") for name in (concepts or {}))
    document = f"{text} {labels}".strip()
    provider = os.getenv("ASAMA_EMBEDDINGS", "local").strip().lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            return openai_embedding(document)
        except Exception:
            # A failed remote call must not take registration down.
            return hashed_embedding(document)
    return hashed_embedding(document)


def shift_embedding(interest: list[float], post: list[float], learning_rate: float) -> list[float]:
    """Move an interest vector toward (positive) or away from (negative) a post."""
    if len(interest) != len(post) or not interest:
        return interest
    mixed = [left + learning_rate * right for left, right in zip(interest, post)]
    return normalize(mixed)
