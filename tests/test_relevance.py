"""Pure routing tests. No database and no network."""

from app.embeddings import cosine, embed_text, jaccard, shift_embedding
from app.ontology import extract_concepts
from app.relevance import THRESHOLD, AgentView, PostView, route_post, score_pair, updated_bias


def _agent(name, manifesto, *, industry="General", active=True, trust=1.0, bias=None, excluded=None, blocks=None, agent_id=None):
    concepts = extract_concepts(manifesto)
    return AgentView(
        id=agent_id or name.lower().replace(" ", "-"),
        company_name=name,
        industry_category=industry,
        manifesto=manifesto,
        embedding=embed_text(manifesto, concepts),
        concepts=concepts,
        bias=bias or {},
        is_active=active,
        excluded_industries=excluded or [],
        blocked_agent_ids=blocks or [],
        trust_score=trust,
    )


def _post(author, title, content, industry="Food Delivery"):
    concepts = extract_concepts(f"{title}\n{content}")
    return PostView(
        id="post-1",
        author_id=author.id,
        author_industry=industry,
        title=title,
        content=content,
        embedding=embed_text(f"{title}\n{content}", concepts),
        concepts=concepts,
    )


DIWALI = (
    "Diwali Food Delivery Discount",
    "Swiggy offering flat 5% off on all delivery orders above ₹300 during Diwali week.",
)


def test_diwali_reaches_payments_and_cards_only():
    swiggy = _agent(
        "Swiggy",
        "Food delivery discounts, restaurant partnerships, festive meal campaigns.",
        industry="Food Delivery",
    )
    phonepe = _agent(
        "PhonePe",
        "Interested in e-commerce promotions, food delivery festive discounts, card payment partnerships, and merchant cashback opportunities.",
        industry="Payments & Fintech",
    )
    hdfc = _agent(
        "HDFC Bank",
        "Credit card partnerships, card payment campaigns, and reward points on merchant spends.",
        industry="Banking & Cards",
    )
    flipkart = _agent(
        "Flipkart",
        "Marketplace promotions across electronics and fashion, including large shopping festivals and seller campaigns.",
        industry="E-commerce",
    )
    bajaj = _agent(
        "Bajaj Auto",
        "Motorcycle launches, service camps, two-wheeler accessories, and vehicle financing for new riders.",
        industry="Automobile",
    )
    post = _post(swiggy, *DIWALI)
    assert jaccard(post.text, hdfc.manifesto) < 0.12
    result = route_post(post, [swiggy, phonepe, hdfc, flipkart, bajaj])
    notified = {item.company_name for item in result.decisions if item.status == "NOTIFIED"}
    assert notified == {"PhonePe", "HDFC Bank"}
    assert result.stage1_excluded == 1
    assert result.notified == 2
    bajaj_row = next(item for item in result.decisions if item.company_name == "Bajaj Auto")
    assert bajaj_row.status == "BELOW_THRESHOLD"
    assert bajaj_row.rerank_score < THRESHOLD
    assert result.timings_ms["total"] < 500


def test_stage1_drops_inactive_blocked_and_excluded():
    author = _agent("Swiggy", "Food delivery discounts and restaurant partnerships.", industry="Food Delivery")
    asleep = _agent("Sleeper", "Credit card partnerships and reward points.", active=False)
    blocked = _agent("Blocked Pay", "UPI payment and merchant checkout partnerships.")
    author.blocked_agent_ids = [blocked.id]
    hates_food = _agent(
        "No Food",
        "Credit card partnerships and card payment campaigns.",
        excluded=["Food Delivery"],
    )
    untrusted = _agent("Shady", "Credit card partnerships and reward points.", trust=0.1)
    post = _post(author, *DIWALI)
    result = route_post(post, [author, asleep, blocked, hates_food, untrusted])
    assert result.decisions == []
    assert result.stage1_eligible == 0


def test_fanout_cap_keeps_five():
    author = _agent("Swiggy", "Food delivery discounts and restaurant partnerships.", industry="Food Delivery")
    partners = [
        _agent(f"Pay {index}", "UPI payment, merchant checkout, and food delivery partnerships.", agent_id=f"pay-{index}")
        for index in range(8)
    ]
    result = route_post(_post(author, *DIWALI), [author, *partners])
    notified = [item for item in result.decisions if item.status == "NOTIFIED"]
    parked = [item for item in result.decisions if item.status == "OVER_CAP"]
    assert len(notified) == 5
    assert len(parked) == 3


def test_repeated_ignores_drop_a_borderline_partner():
    affinity = 0.90
    vector = 0.2
    bias: dict[str, float] = {}
    opening = score_pair(affinity, vector, 1.0, bias, "food_delivery")
    assert opening >= THRESHOLD
    for _ in range(6):
        bias = updated_bias(bias, "food_delivery", -0.05)
    assert score_pair(affinity, vector, 1.0, bias, "food_delivery") < THRESHOLD


def test_negative_shift_moves_away_from_the_post():
    post = embed_text("food delivery diwali discount", {"food_delivery": 1.0, "festive": 0.5})
    other = embed_text("motorcycle service camp financing", {"auto": 1.0, "lending": 0.5})
    interest = shift_embedding(post, other, 0.4)
    before = cosine(interest, post)
    after = cosine(shift_embedding(interest, post, -0.5), post)
    assert after < before


def test_shortlist_keeps_a_complementary_agent_among_noise():
    author = _agent("Swiggy", "Food delivery discounts and restaurant partnerships.", industry="Food Delivery")
    cards = _agent("CardsCo", "Credit card partnerships, card payment campaigns, and reward points.")
    noise = [
        _agent(f"Noise {index}", f"Quarterly note about office plants and team offsites batch {index}.", agent_id=f"n-{index}")
        for index in range(80)
    ]
    result = route_post(_post(author, *DIWALI), [author, cards, *noise])
    match = next(item for item in result.decisions if item.company_name == "CardsCo")
    assert match.status == "NOTIFIED"
    assert result.timings_ms["total"] < 1000
