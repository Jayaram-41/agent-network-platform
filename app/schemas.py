"""Request and response bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRegister(BaseModel):
    company_name: str = Field(min_length=2, max_length=255)
    industry_category: str = Field(min_length=2, max_length=100)
    webhook_url: str = Field(default="", max_length=2000)
    interest_manifesto: str = Field(min_length=20, max_length=5000)
    excluded_industries: list[str] = Field(default_factory=list)
    playbooks: list[dict] = Field(default_factory=list)


class AgentCreated(BaseModel):
    id: str
    company_name: str
    industry_category: str
    api_key: str
    webhook_url: str
    trust_score: float


class AgentPublic(BaseModel):
    id: str
    company_name: str
    industry_category: str
    interest_manifesto: str
    trust_score: float
    is_active: bool
    webhook_url: str
    preference_bias: dict[str, float]


class PostCreate(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    content: str = Field(min_length=10, max_length=8000)
    author_agent_id: str | None = None
    expires_at: str | None = None


class CommentCreate(BaseModel):
    content: str = Field(min_length=2, max_length=4000)
    author_agent_id: str | None = None


class FeedbackCreate(BaseModel):
    decision: str = Field(pattern="^(ignore|engage)$")


class AgentBrief(BaseModel):
    id: str
    company_name: str
    industry_category: str
    trust_score: float


class PostBody(BaseModel):
    id: str
    title: str
    content: str
    status: str
    safety_reasons: list[str]
    created_at: str
    author: AgentBrief


class DecisionOut(BaseModel):
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


class RoutingOut(BaseModel):
    threshold: float
    timings_ms: dict[str, float]
    stage1_eligible: int
    stage1_excluded: int
    notified: int
    primary_topic: str
    decisions: list[DecisionOut]


class CommentOut(BaseModel):
    id: str
    author_agent_id: str
    company_name: str
    content: str
    created_at: str


class PublishResult(BaseModel):
    post: PostBody
    routing: RoutingOut | None
    comments: list[CommentOut]


class MetricsOut(BaseModel):
    agents: int
    posts: int
    notifications: int
    engagements: int
    engagement_rate: float


class DemoAgent(BaseModel):
    id: str
    company_name: str
    industry_category: str
    api_key: str


class DemoSession(BaseModel):
    demo_mode: bool
    agents: list[DemoAgent]
    note: str
