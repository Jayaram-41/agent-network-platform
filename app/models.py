"""Relational model. Embeddings are JSON arrays so SQLite can store them.

The production DDL in schema.sql uses pgvector instead of JSON text.
"""

from __future__ import annotations

import json

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    industry_category: Mapped[str] = mapped_column(String(100), nullable=False)
    webhook_url: Mapped[str] = mapped_column(Text, nullable=False)
    webhook_secret: Mapped[str] = mapped_column(String(128), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    interest_manifesto: Mapped[str] = mapped_column(Text, nullable=False)
    interest_embedding: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    concepts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    preference_bias: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    playbooks: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    excluded_industries: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    blocked_agent_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    trust_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)

    posts: Mapped[list["Post"]] = relationship(back_populates="author", foreign_keys="Post.author_agent_id")

    @property
    def playbooks_data(self) -> list:
        return json.loads(self.playbooks or "[]")


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    author_agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    post_embedding: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    concepts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="published")
    safety_reasons: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    expires_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    routing_meta: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)

    author: Mapped[Agent] = relationship(back_populates="posts", foreign_keys=[author_agent_id])
    comments: Mapped[list["Comment"]] = relationship(back_populates="post")
    routing_logs: Mapped[list["RoutingLog"]] = relationship(back_populates="post")


class RoutingLog(Base):
    __tablename__ = "routing_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    target_agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    vector_score: Mapped[float] = mapped_column(Float, nullable=False)
    recall_score: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="NOTIFIED")
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    primary_topic: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    matched_interest: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    feedback_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)

    post: Mapped[Post] = relationship(back_populates="routing_logs")
    target: Mapped[Agent] = relationship()


class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = (UniqueConstraint("post_id", "author_agent_id", name="uq_comment_author"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    author_agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)

    post: Mapped[Post] = relationship(back_populates="comments")
    author: Mapped[Agent] = relationship()
