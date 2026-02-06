"""Centralised configuration — reads environment variables with sensible defaults.

All backend modules should ``from backend.config import ...`` instead of
reading ``os.environ`` directly.
"""

from __future__ import annotations

import os
import secrets

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- OpenAI ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
MAX_ITERS: int = int(os.environ.get("MAX_ITERS", "2"))

# --- Paths ---
UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads"))
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"))
MDIMG_DB_PATH: str = os.environ.get("MDIMG_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mdimg.db"))

# --- Flask ---
SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_hex(32))
FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50 MB


def apply_to_env() -> None:
    """Push config values into ``os.environ`` so the pipeline package picks
    them up without modification (it reads ``OPENAI_API_KEY``, ``OPENAI_MODEL``,
    ``MDIMG_DB_PATH`` from env)."""
    if OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    os.environ["OPENAI_MODEL"] = OPENAI_MODEL
    os.environ["MDIMG_DB_PATH"] = MDIMG_DB_PATH
