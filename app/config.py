"""Runtime settings. Environment variables override the local demo defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path
    demo_mode: bool
    seed: bool
    use_llm: bool
    embedding_provider: str
    openai_model: str
    threshold: float = 0.75


def load_settings() -> Settings:
    _load_dotenv()
    data_dir = Path(os.getenv("ASAMA_DATA_DIR", "./data"))
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./data/asama.db"),
        data_dir=data_dir,
        demo_mode=_as_bool("ASAMA_DEMO_MODE", True),
        seed=_as_bool("ASAMA_SEED", True),
        use_llm=_as_bool("ASAMA_USE_LLM", False),
        embedding_provider=os.getenv("ASAMA_EMBEDDINGS", "local").strip().lower(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
