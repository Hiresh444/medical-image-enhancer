"""SQLite persistence layer for pipeline runs, chat messages, and agent traces.

Uses raw ``sqlite3`` (no ORM) for zero extra dependencies.  DB location is
configurable via the ``MDIMG_DB_PATH`` environment variable (default:
``data/mdimg.db`` relative to the project root).
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

_DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "mdimg.db")


def _db_path() -> str:
    return os.environ.get("MDIMG_DB_PATH", _DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    input_filename  TEXT NOT NULL,
    metadata_summary TEXT DEFAULT '{}',
    issues          TEXT DEFAULT '[]',
    metrics_before  TEXT DEFAULT '{}',
    metrics_after   TEXT DEFAULT '{}',
    plan_json       TEXT DEFAULT '',
    validation      TEXT DEFAULT '{}',
    applied_ops     TEXT DEFAULT '[]',
    explainability  TEXT DEFAULT '{}',
    report_path     TEXT DEFAULT '',
    before_after_path TEXT DEFAULT '',
    agent_logs      TEXT DEFAULT '[]',
    status          TEXT DEFAULT 'completed',
    genai_model     TEXT DEFAULT '',
    genai_llm_calls INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_run ON chat_messages(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(timestamp);
"""


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return uuid.uuid4().hex[:12]


def insert_pending_run(run_id: str, input_filename: str) -> None:
    """Insert a minimal row with status='pending' so polling can begin."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO runs (run_id, timestamp, input_filename, status)
               VALUES (?, ?, ?, ?)""",
            (run_id, datetime.now(timezone.utc).isoformat(), input_filename, "pending"),
        )
        conn.commit()
    finally:
        conn.close()


def update_run_status(run_id: str, status: str) -> None:
    """Update only the status column for a run."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id)
        )
        conn.commit()
    finally:
        conn.close()


def save_run(
    run_id: str,
    input_filename: str,
    metadata_summary: dict,
    issues: list[str],
    metrics_before: dict,
    metrics_after: dict,
    plan_json: str,
    validation: dict,
    applied_ops: list[str],
    explainability: dict | str,
    report_path: str,
    before_after_path: str,
    agent_logs: list[dict],
    status: str = "completed",
    genai_model: str = "",
    genai_llm_calls: int = 0,
) -> None:
    """Insert a completed run into the database."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, timestamp, input_filename, metadata_summary, issues,
                metrics_before, metrics_after, plan_json, validation,
                applied_ops, explainability, report_path, before_after_path,
                agent_logs, status, genai_model, genai_llm_calls)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                input_filename,
                json.dumps(metadata_summary, default=str),
                json.dumps(issues),
                json.dumps(_serialise(metrics_before)),
                json.dumps(_serialise(metrics_after)),
                plan_json,
                json.dumps(_serialise(validation)),
                json.dumps(applied_ops),
                json.dumps(explainability, default=str) if isinstance(explainability, dict) else str(explainability),
                report_path,
                before_after_path,
                json.dumps(agent_logs, default=str),
                status,
                genai_model,
                genai_llm_calls,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    """Fetch a single run by ID.  Returns None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def list_runs(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """List runs ordered by most recent first."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


def save_chat_message(
    run_id: str, role: str, content: str
) -> None:
    """Append a chat message for a run."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO chat_messages (run_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (run_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_chat_history(run_id: str) -> list[dict[str, str]]:
    """Get all chat messages for a run, ordered chronologically."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM chat_messages WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, JSON-parsing stored fields."""
    d = dict(row)
    for key in (
        "metadata_summary", "issues", "metrics_before", "metrics_after",
        "validation", "applied_ops", "agent_logs",
    ):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    # Parse explainability (may be JSON or plain text)
    if "explainability" in d and isinstance(d["explainability"], str):
        try:
            d["explainability"] = json.loads(d["explainability"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _serialise(obj: Any) -> Any:
    """Make numpy/bool types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
