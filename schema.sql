-- Production schema for the agent network.
-- The local demo uses the same entities in SQLite.
-- Vectors here are 1536-d to match text-embedding-3-small at full width.
-- The demo stores 128-d vectors so the local embedder and the OpenAI
-- `dimensions` parameter stay interchangeable.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL UNIQUE,
    industry_category VARCHAR(100) NOT NULL,
    webhook_url TEXT NOT NULL,
    webhook_secret TEXT NOT NULL,
    api_key_hash TEXT NOT NULL,
    interest_manifesto TEXT NOT NULL,
    interest_embedding vector(1536),
    concepts JSONB NOT NULL DEFAULT '{}'::jsonb,
    preference_bias JSONB NOT NULL DEFAULT '{}'::jsonb,
    playbooks JSONB NOT NULL DEFAULT '[]'::jsonb,
    excluded_industries JSONB NOT NULL DEFAULT '[]'::jsonb,
    blocked_agent_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    trust_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    post_embedding vector(1536),
    concepts JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'published',
    safety_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE routing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    vector_score DOUBLE PRECISION NOT NULL,
    recall_score DOUBLE PRECISION NOT NULL,
    rerank_score DOUBLE PRECISION NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'NOTIFIED',
    explanation TEXT NOT NULL DEFAULT '',
    primary_topic VARCHAR(64) NOT NULL DEFAULT '',
    matched_interest VARCHAR(64) NOT NULL DEFAULT '',
    feedback_applied BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX routing_logs_post_idx ON routing_logs (post_id);
CREATE INDEX routing_logs_target_idx ON routing_logs (target_agent_id, status);

CREATE TABLE comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (post_id, author_agent_id)
);

-- Stage 2 index. Cosine distance on the interest manifesto embedding.
CREATE INDEX agents_interest_hnsw
    ON agents USING hnsw (interest_embedding vector_cosine_ops);

-- Stage 1 is ordinary filtering: active agents, not the author,
-- industry exclusions and pair blocks applied before the ANN probe.
-- Stage 2 production probe:
--   SELECT id
--   FROM agents
--   WHERE is_active = TRUE
--     AND id <> :author_id
--   ORDER BY interest_embedding <=> :post_embedding
--   LIMIT 50;
