"""API keys and webhook signatures."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from urllib.parse import urlparse


def new_api_key() -> str:
    return "ak_" + secrets.token_urlsafe(24)


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def dumps(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(secret: str, body: bytes, signature: str) -> bool:
    expected = sign(secret, body)
    return hmac.compare_digest(expected, signature)


def validate_webhook(url: str) -> None:
    if url.startswith("internal://"):
        return
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return
    if parsed.scheme == "http" and host in {"127.0.0.1", "localhost"}:
        return
    raise ValueError("webhook_url must be https, localhost http, or an internal:// agent")
