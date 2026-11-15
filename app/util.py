"""Small shared helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())
