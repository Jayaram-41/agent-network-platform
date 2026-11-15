"""Business-topic ontology used by the relevance router.

Keyword overlap cannot see that a Diwali food discount matters to a card
issuer. This map encodes that complementarity explicitly. It is the feature
a cross-encoder would learn from engagement data; the weights are the
starting prior, and the feedback loop nudges them per agent.
"""

from __future__ import annotations

import re

# Occasion and incentive language modifies a post. It is not a business line.
MODIFIERS = {"festive", "promotion"}

# When two domain topics tie, keep the operating business rather than the rail
# it happens to mention ("Swiggy ... UPI" is still a food-delivery post).
PRIORITY = [
    "food_delivery",
    "auto",
    "insurance",
    "logistics",
    "lending",
    "cards",
    "payments",
    "ecommerce",
]

KEYWORDS: dict[str, tuple[str, ...]] = {
    "food_delivery": (
        "food delivery",
        "swiggy",
        "zomato",
        "restaurant",
        "delivery orders",
        "meal",
    ),
    "ecommerce": (
        "e-commerce",
        "ecommerce",
        "flipkart",
        "marketplace",
        "electronics",
        "fashion",
    ),
    "auto": (
        "motorcycle",
        "two-wheeler",
        "two wheeler",
        "vehicle",
        "scooter",
        "automobile",
        "service camp",
    ),
    "payments": (
        "upi",
        "phonepe",
        "payment",
        "payments",
        "wallet",
        "checkout",
        "merchant",
    ),
    "cards": (
        "credit card",
        "debit card",
        "card payment",
        "reward points",
        "cardholder",
    ),
    "lending": ("loan", "financing", "emi", "nbfc"),
    "insurance": ("insurance", "premium", "policyholder"),
    "logistics": ("logistics", "warehouse", "courier", "last mile"),
    "festive": ("diwali", "holi", "navratri", "christmas", "festive", "festival", "festivals"),
    "promotion": ("discount", "cashback", "offer", "offers", "sale"),
}

# Directed complementarity. pair_affinity reads both directions.
AFFINITY: dict[str, dict[str, float]] = {
    "food_delivery": {
        "payments": 0.95,
        "cards": 0.90,
        "logistics": 0.62,
        "ecommerce": 0.25,
        "lending": 0.18,
        "insurance": 0.12,
        "auto": 0.05,
    },
    "ecommerce": {
        "payments": 0.93,
        "cards": 0.88,
        "logistics": 0.80,
        "lending": 0.30,
        "food_delivery": 0.25,
        "auto": 0.08,
    },
    "auto": {
        "insurance": 0.86,
        "lending": 0.84,
        "cards": 0.28,
        "payments": 0.22,
        "ecommerce": 0.12,
        "food_delivery": 0.04,
    },
    "payments": {"cards": 0.90, "lending": 0.45},
    "cards": {"lending": 0.55},
    "lending": {"insurance": 0.35},
}


def pair_affinity(source: str, target: str) -> float:
    """How strongly a post about `source` should reach an agent interested in `target`."""
    if source == target:
        return 1.0
    forward = AFFINITY.get(source, {}).get(target, 0.0)
    backward = AFFINITY.get(target, {}).get(source, 0.0)
    return max(forward, backward)


def _contains(text: str, phrase: str) -> bool:
    if " " in phrase or not phrase.isalnum():
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def extract_concepts(text: str) -> dict[str, float]:
    """Return concept -> strength in (0, 1]. One clear keyword is enough to fire."""
    lowered = text.lower()
    found: dict[str, float] = {}
    for concept, phrases in KEYWORDS.items():
        hits = sum(1 for phrase in phrases if _contains(lowered, phrase))
        if hits:
            found[concept] = min(1.0, 0.5 + 0.25 * (hits - 1))
    return found


def primary_topic(concepts: dict[str, float]) -> str | None:
    domains = {name: score for name, score in concepts.items() if name not in MODIFIERS}
    if not domains:
        return None
    best = max(domains.values())
    contenders = [name for name, score in domains.items() if score == best]
    contenders.sort(key=lambda name: PRIORITY.index(name) if name in PRIORITY else len(PRIORITY))
    return contenders[0]


def domain_concepts(concepts: dict[str, float]) -> list[str]:
    return [name for name in concepts if name not in MODIFIERS]
