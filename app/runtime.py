"""Company-agent runtime.

A notified agent reads the webhook as data, decides, and either drafts a
comment or stays silent. Playbooks cover the demo without an API key.
ASAMA_USE_LLM=true swaps the draft step for a chat completion that still
receives the post only as a JSON user message.
"""

from __future__ import annotations

import os

from app.ontology import extract_concepts
from app.safety import build_llm_messages, comment_is_safe, scan


def select_playbook(playbooks: list[dict], concepts: dict[str, float]) -> dict | None:
    matching = [item for item in playbooks if all(name in concepts for name in item.get("all", []))]
    if not matching:
        return None
    matching.sort(key=lambda item: len(item.get("all", [])), reverse=True)
    return matching[0]


def draft_with_llm(company: str, manifesto: str, post: dict, model: str) -> str | None:
    import httpx

    messages = build_llm_messages(company, manifesto, post)
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model, "temperature": 0.2, "messages": messages},
        timeout=20.0,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    if not text or text.upper() == "DECLINE" or not comment_is_safe(text):
        return None
    return text


def decide(agent, payload: dict, *, use_llm: bool = False, model: str = "gpt-4o-mini") -> str | None:
    """Return a comment, or None when the agent declines."""
    post = payload.get("post") or {}
    text = f"{post.get('title', '')}\n{post.get('content', '')}"
    if scan(text).blocked:
        return None
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            drafted = draft_with_llm(agent.company_name, agent.interest_manifesto, post, model)
            if drafted:
                return drafted
        except Exception:
            pass
    concepts = extract_concepts(text)
    playbook = select_playbook(list(agent.playbooks_data), concepts)
    if playbook is None:
        return None
    author = post.get("author_company") or "the partner"
    return str(playbook["template"]).format(author=author, company=agent.company_name)
