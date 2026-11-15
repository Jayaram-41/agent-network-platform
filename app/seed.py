"""Seed companies for the Diwali walkthrough.

Demo API keys are fixed so a reviewer can curl the scenario. They exist only
when the demo seed runs. Turn ASAMA_DEMO_MODE and ASAMA_SEED off for anything
else, and issue keys from /api/v1/agents/register.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_text
from app.models import Agent
from app.ontology import extract_concepts
from app.security import hash_key
from app.util import utcnow

DEMO_KEYS = {
    "Swiggy": "demo-swiggy-key",
    "PhonePe": "demo-phonepe-key",
    "HDFC Bank": "demo-hdfc-key",
    "Flipkart": "demo-flipkart-key",
    "Bajaj Auto": "demo-bajaj-key",
}

# Stable ids. Swiggy uses the sample id from the assignment brief.
DEMO_IDS = {
    "Swiggy": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "PhonePe": "7c2ceb4d-1a2b-3c4d-5e6f-7a8b9c0d1e2f",
    "HDFC Bank": "11111111-2222-4333-8444-555555555555",
    "Flipkart": "22222222-3333-4444-8555-666666666666",
    "Bajaj Auto": "33333333-4444-4555-8666-777777777777",
}

SEED_AGENTS = [
    {
        "company_name": "Swiggy",
        "industry_category": "Food Delivery",
        "interest_manifesto": (
            "Food delivery discounts, restaurant partnerships, festive meal campaigns, "
            "and lower delivery fees on partner brands."
        ),
        "playbooks": [
            {
                "all": ["payments"],
                "template": (
                    "Swiggy can run a co-branded checkout banner for {author} "
                    "across food delivery orders this week."
                ),
            },
            {
                "all": ["cards"],
                "template": "Swiggy can add a delivery benefit when customers pay with {author}.",
            },
        ],
    },
    {
        "company_name": "PhonePe",
        "industry_category": "Payments & Fintech",
        "interest_manifesto": (
            "Interested in e-commerce promotions, food delivery festive discounts, "
            "card payment partnerships, and merchant cashback opportunities."
        ),
        "playbooks": [
            {
                "all": ["food_delivery", "festive"],
                "template": (
                    "Extra ₹50 cashback when you pay for {author} orders with PhonePe UPI on Diwali."
                ),
            },
            {
                "all": ["food_delivery"],
                "template": "Extra ₹50 cashback when you pay for {author} orders with PhonePe UPI.",
            },
            {
                "all": ["ecommerce"],
                "template": "PhonePe will add UPI cashback on {author} checkouts during this sale.",
            },
        ],
    },
    {
        "company_name": "HDFC Bank",
        "industry_category": "Banking & Cards",
        "interest_manifesto": (
            "Credit card partnerships, card payment campaigns, and reward points on merchant spends."
        ),
        "playbooks": [
            {
                "all": ["food_delivery"],
                "template": (
                    "HDFC Bank credit cards will give 10% extra reward points "
                    "when customers pay for {author} orders during this promotion."
                ),
            },
            {
                "all": ["ecommerce"],
                "template": (
                    "HDFC Bank will offer an instant discount on credit card checkout for {author}'s campaign."
                ),
            },
        ],
    },
    {
        "company_name": "Flipkart",
        "industry_category": "E-commerce",
        "interest_manifesto": (
            "Marketplace promotions across electronics and fashion, including large "
            "shopping festivals and seller campaigns."
        ),
        "playbooks": [
            {
                "all": ["payments"],
                "template": "Flipkart can feature {author} as a checkout partner for the upcoming sale window.",
            }
        ],
    },
    {
        "company_name": "Bajaj Auto",
        "industry_category": "Automobile",
        "interest_manifesto": (
            "Motorcycle launches, service camps, two-wheeler accessories, "
            "and vehicle financing for new riders."
        ),
        "playbooks": [
            {
                "all": ["lending"],
                "template": (
                    "Bajaj Auto can bundle a financing offer with {author} for riders buying a new two-wheeler."
                ),
            },
            {
                "all": ["insurance"],
                "template": "Bajaj Auto can include {author} cover as an add-on at vehicle delivery.",
            },
        ],
    },
]


def _build_agent(spec: dict) -> Agent:
    name = spec["company_name"]
    concepts = extract_concepts(spec["interest_manifesto"])
    embedding = embed_text(spec["interest_manifesto"], concepts)
    return Agent(
        id=DEMO_IDS[name],
        company_name=name,
        industry_category=spec["industry_category"],
        webhook_url=f"internal://agents/{DEMO_IDS[name]}",
        webhook_secret=f"whsec-{name.lower().replace(' ', '-')}",
        api_key_hash=hash_key(DEMO_KEYS[name]),
        interest_manifesto=spec["interest_manifesto"],
        interest_embedding=json.dumps(embedding),
        concepts=json.dumps(concepts),
        preference_bias="{}",
        playbooks=json.dumps(spec["playbooks"]),
        excluded_industries="[]",
        blocked_agent_ids="[]",
        trust_score=1.0,
        is_active=True,
        created_at=utcnow(),
    )


def seed_if_empty(db: Session, data_dir: Path) -> None:
    existing = db.scalar(select(Agent.id).limit(1))
    if existing:
        return
    for spec in SEED_AGENTS:
        db.add(_build_agent(spec))
    db.flush()
    write_demo_keys(data_dir)


def write_demo_keys(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "Local demo keys. Disable ASAMA_DEMO_MODE outside this exercise.",
        "agents": [
            {
                "id": DEMO_IDS[spec["company_name"]],
                "company_name": spec["company_name"],
                "industry_category": spec["industry_category"],
                "api_key": DEMO_KEYS[spec["company_name"]],
            }
            for spec in SEED_AGENTS
        ],
    }
    (data_dir / "demo_keys.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_demo_keys(data_dir: Path) -> dict:
    path = data_dir / "demo_keys.json"
    if not path.exists():
        return {"agents": []}
    return json.loads(path.read_text(encoding="utf-8"))
