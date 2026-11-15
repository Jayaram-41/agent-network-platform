"""Three-stage relevance routing.

Stage 1 drops agents with a SQL-shaped metadata rule: inactive, the author,
blocked pairs, excluded industries, and agents whose trust has collapsed.
Stage 2 ranks the rest by a hybrid recall score (dense cosine, topic
affinity, lexical overlap) and keeps the top K, the stand-in for an HNSW
probe. Stage 3 rescores those candidates and notifies at most five whose
final score clears 0.75.

The final score is dominated by business-topic affinity. Dense cosine is a
small tie-break so a weak embedding cannot drown a real partnership, and a
shared adjective cannot create one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.embeddings import cosine, jaccard
from app.ontology import domain_concepts, pair_affinity, primary_topic

THRESHOLD = 0.75
ANN_K = 50
MAX_NOTIFY = 5
IGNORE_DELTA = -0.05
ENGAGE_DELTA = 0.04


@dataclass
class AgentView:
    id: str
    company_name: str
    industry_category: str
    manifesto: str
    embedding: list[float]
    concepts: dict[str, float]
    bias: dict[str, float]
    is_active: bool
    excluded_industries: list[str]
    blocked_agent_ids: list[str]
    trust_score: float


@dataclass
class PostView:
    id: str
    author_id: str
    author_industry: str
    title: str
    content: str
    embedding: list[float]
    concepts: dict[str, float]

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"


@dataclass
class Decision:
    agent_id: str
    company_name: str
    industry_category: str
    vector_score: float
    recall_score: float
    rerank_score: float
    status: str
    explanation: str
    primary_topic: str
    matched_interest: str
    affinity: float


@dataclass
class RouteResult:
    decisions: list[Decision]
    timings_ms: dict[str, float]
    stage1_eligible: int
    stage1_excluded: int
    notified: int
    threshold: float = THRESHOLD
    primary_topic: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def affinity_match(post_concepts: dict[str, float], agent_concepts: dict[str, float]) -> tuple[float, str, str]:
    topic = primary_topic(post_concepts) or ""
    interests = domain_concepts(agent_concepts)
    if not topic or not interests:
        return 0.0, topic, ""
    matched = max(interests, key=lambda interest: pair_affinity(topic, interest))
    return pair_affinity(topic, matched), topic, matched


def score_pair(
    affinity: float,
    vector_score: float,
    trust_score: float,
    bias: dict[str, float],
    topic: str,
) -> float:
    base = min(1.0, 0.9 * affinity + 0.1 * max(vector_score, 0.0))
    trust_weight = 0.9 + 0.1 * _clamp(trust_score, 0.0, 1.0)
    bias_weight = _clamp(1.0 + float(bias.get(topic, 0.0) if topic else 0.0), 0.2, 1.3)
    return _clamp(base * trust_weight * bias_weight, 0.0, 1.0)


def updated_bias(bias: dict[str, float], topic: str, delta: float) -> dict[str, float]:
    if not topic:
        return dict(bias)
    updated = dict(bias)
    updated[topic] = _clamp(float(updated.get(topic, 0.0)) + delta, -0.8, 0.5)
    return updated


def _explain(decision_status: str, topic: str, matched: str, affinity: float, vector_score: float, final: float) -> str:
    if not topic:
        return (
            f"No business topic was recognised in the post. "
            f"Dense cosine is {vector_score:.2f} and the final score is {final:.2f}."
        )
    relation = "matches" if topic == matched else "complements"
    if decision_status == "NOTIFIED":
        outcome = f"Final score {final:.2f} clears the {THRESHOLD:.2f} cutoff."
    elif decision_status == "OVER_CAP":
        outcome = (
            f"Final score {final:.2f} clears the cutoff, but the fan-out cap of "
            f"{MAX_NOTIFY} higher-scoring agents is already full."
        )
    else:
        outcome = f"Final score {final:.2f} is under the {THRESHOLD:.2f} cutoff, so this agent is not notified."
    return (
        f"Primary topic is {topic.replace('_', ' ')}. "
        f"It {relation} this agent's interest in {matched.replace('_', ' ') if matched else 'nothing on file'} "
        f"(affinity {affinity:.2f}, dense cosine {vector_score:.2f}). {outcome}"
    )


def stage1(post: PostView, agents: list[AgentView]) -> tuple[list[AgentView], int]:
    author = next((agent for agent in agents if agent.id == post.author_id), None)
    author_blocks = set(author.blocked_agent_ids) if author else set()
    eligible: list[AgentView] = []
    excluded = 0
    for agent in agents:
        if not agent.is_active or agent.id == post.author_id:
            excluded += 1
            continue
        if agent.trust_score < 0.25:
            excluded += 1
            continue
        if post.author_industry in agent.excluded_industries:
            excluded += 1
            continue
        if agent.id in author_blocks or post.author_id in agent.blocked_agent_ids:
            excluded += 1
            continue
        eligible.append(agent)
    return eligible, excluded


def route_post(post: PostView, agents: list[AgentView]) -> RouteResult:
    started = time.perf_counter()
    eligible, excluded = stage1(post, agents)
    stage1_done = time.perf_counter()

    ranked: list[tuple[float, float, float, str, str, AgentView]] = []
    for agent in eligible:
        affinity, topic, matched = affinity_match(post.concepts, agent.concepts)
        vector_score = cosine(post.embedding, agent.embedding)
        lexical = jaccard(post.text, agent.manifesto)
        # Affinity dominates recall so a crowd of weakly related agents cannot
        # push a real partner out of the top-K shortlist.
        recall = 0.82 * affinity + 0.12 * max(vector_score, 0.0) + 0.06 * lexical
        ranked.append((recall, affinity, vector_score, topic, matched, agent))
    ranked.sort(key=lambda item: item[0], reverse=True)
    shortlist = ranked[:ANN_K]
    stage2_done = time.perf_counter()

    rescored: list[Decision] = []
    for recall, affinity, vector_score, topic, matched, agent in shortlist:
        final = score_pair(affinity, vector_score, agent.trust_score, agent.bias, topic)
        status = "NOTIFIED" if final >= THRESHOLD else "BELOW_THRESHOLD"
        rescored.append(
            Decision(
                agent_id=agent.id,
                company_name=agent.company_name,
                industry_category=agent.industry_category,
                vector_score=round(vector_score, 4),
                recall_score=round(recall, 4),
                rerank_score=round(final, 4),
                status=status,
                explanation="",
                primary_topic=topic,
                matched_interest=matched,
                affinity=round(affinity, 4),
            )
        )
    rescored.sort(key=lambda item: item.rerank_score, reverse=True)

    notified = 0
    for decision in rescored:
        if decision.status != "NOTIFIED":
            continue
        if notified >= MAX_NOTIFY:
            decision.status = "OVER_CAP"
            continue
        notified += 1

    for decision in rescored:
        decision.explanation = _explain(
            decision.status,
            decision.primary_topic,
            decision.matched_interest,
            decision.affinity,
            decision.vector_score,
            decision.rerank_score,
        )

    finished = time.perf_counter()
    topic = primary_topic(post.concepts) or ""
    return RouteResult(
        decisions=rescored,
        timings_ms={
            "stage1": round((stage1_done - started) * 1000, 3),
            "stage2": round((stage2_done - stage1_done) * 1000, 3),
            "stage3": round((finished - stage2_done) * 1000, 3),
            "total": round((finished - started) * 1000, 3),
        },
        stage1_eligible=len(eligible),
        stage1_excluded=excluded,
        notified=notified,
        primary_topic=topic,
    )
