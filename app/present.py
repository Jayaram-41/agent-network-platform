"""Map database rows onto API models."""

from __future__ import annotations

import json

from app.models import Agent, Comment, Post, RoutingLog
from app.schemas import (
    AgentBrief,
    AgentPublic,
    CommentOut,
    DecisionOut,
    PostBody,
    PublishResult,
    RoutingOut,
)
from app.services import SENT_STATUSES


def _loads(raw: str, fallback):
    try:
        return json.loads(raw) if raw else fallback
    except json.JSONDecodeError:
        return fallback


def agent_public(agent: Agent) -> AgentPublic:
    return AgentPublic(
        id=agent.id,
        company_name=agent.company_name,
        industry_category=agent.industry_category,
        interest_manifesto=agent.interest_manifesto,
        trust_score=round(agent.trust_score, 4),
        is_active=agent.is_active,
        webhook_url=agent.webhook_url,
        preference_bias=_loads(agent.preference_bias, {}),
    )


def post_body(post: Post) -> PostBody:
    author = post.author
    return PostBody(
        id=post.id,
        title=post.title,
        content=post.content,
        status=post.status,
        safety_reasons=_loads(post.safety_reasons, []),
        created_at=post.created_at,
        author=AgentBrief(
            id=author.id,
            company_name=author.company_name,
            industry_category=author.industry_category,
            trust_score=round(author.trust_score, 4),
        ),
    )


def comment_out(comment: Comment) -> CommentOut:
    return CommentOut(
        id=comment.id,
        author_agent_id=comment.author_agent_id,
        company_name=comment.author.company_name,
        content=comment.content,
        created_at=comment.created_at,
    )


def routing_from_result(post: Post, result) -> RoutingOut:
    return RoutingOut(
        threshold=result.threshold,
        timings_ms=result.timings_ms,
        stage1_eligible=result.stage1_eligible,
        stage1_excluded=result.stage1_excluded,
        notified=sum(1 for item in result.decisions if item.status in SENT_STATUSES),
        primary_topic=result.primary_topic,
        decisions=[
            DecisionOut(
                agent_id=item.agent_id,
                company_name=item.company_name,
                industry_category=item.industry_category,
                vector_score=item.vector_score,
                recall_score=item.recall_score,
                rerank_score=item.rerank_score,
                status=item.status,
                explanation=item.explanation,
                primary_topic=item.primary_topic,
                matched_interest=item.matched_interest,
            )
            for item in result.decisions
        ],
    )


def routing_from_logs(post: Post) -> RoutingOut | None:
    logs: list[RoutingLog] = list(post.routing_logs or [])
    if not logs and post.status != "published":
        return None
    if not logs:
        return None
    meta = _loads(post.routing_meta, {})
    decisions = []
    for log in sorted(logs, key=lambda item: item.rerank_score, reverse=True):
        target = log.target
        decisions.append(
            DecisionOut(
                agent_id=log.target_agent_id,
                company_name=target.company_name,
                industry_category=target.industry_category,
                vector_score=log.vector_score,
                recall_score=log.recall_score,
                rerank_score=log.rerank_score,
                status=log.status,
                explanation=log.explanation,
                primary_topic=log.primary_topic,
                matched_interest=log.matched_interest,
            )
        )
    return RoutingOut(
        threshold=float(meta.get("threshold", 0.75)),
        timings_ms=meta.get("timings_ms") or {},
        stage1_eligible=int(meta.get("stage1_eligible", 0)),
        stage1_excluded=int(meta.get("stage1_excluded", 0)),
        notified=sum(1 for item in decisions if item.status in SENT_STATUSES),
        primary_topic=str(meta.get("primary_topic", decisions[0].primary_topic if decisions else "")),
        decisions=decisions,
    )


def publish_result(post: Post, result, comments: list[Comment]) -> PublishResult:
    routing = routing_from_result(post, result) if result is not None else routing_from_logs(post)
    return PublishResult(post=post_body(post), routing=routing, comments=[comment_out(item) for item in comments])
