"""Agent trace logger — captures prompts, tool calls, and iteration summaries.

Structured logs are stored per-run and can be persisted to SQLite.
All content is sanitised to exclude PHI before storage.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# Regex to strip potential PHI patterns (patient names, IDs)
_PHI_PATTERN = re.compile(
    r"(?i)(patient\s*(name|id|dob|birth|ssn)\s*[:=]\s*\S+)", re.IGNORECASE
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitise_text(text: str, max_len: int = 2000) -> str:
    """Remove control characters, potential PHI patterns, and truncate."""
    text = _CTRL_RE.sub("", text)
    text = _PHI_PATTERN.sub("[REDACTED]", text)
    return text[:max_len]


@dataclass
class TraceEntry:
    """A single trace log entry."""

    timestamp: float
    phase: str
    event: str  # "phase_start", "prompt", "tool_call", "iteration", "phase_end"
    detail: dict[str, Any] = field(default_factory=dict)


class AgentTraceLogger:
    """Collects structured agent trace entries for a single pipeline run."""

    def __init__(self) -> None:
        self.entries: list[TraceEntry] = []
        self._current_phase: str = ""

    def log_phase_start(self, phase_name: str) -> None:
        self._current_phase = phase_name
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=phase_name,
                event="phase_start",
            )
        )

    def log_phase_end(self, phase_name: str, outcome: str = "") -> None:
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=phase_name,
                event="phase_end",
                detail={"outcome": _sanitise_text(outcome)},
            )
        )

    def log_prompt(self, prompt_summary: str) -> None:
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=self._current_phase,
                event="prompt",
                detail={"summary": _sanitise_text(prompt_summary)},
            )
        )

    def log_tool_call(
        self,
        tool_name: str,
        args_summary: str = "",
        result_summary: str = "",
    ) -> None:
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=self._current_phase,
                event="tool_call",
                detail={
                    "tool": tool_name,
                    "args": _sanitise_text(args_summary, max_len=500),
                    "result": _sanitise_text(result_summary, max_len=500),
                },
            )
        )

    def log_iteration(
        self,
        iteration_num: int,
        score: float,
        metrics_summary: dict[str, float] | None = None,
    ) -> None:
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=self._current_phase,
                event="iteration",
                detail={
                    "iteration": iteration_num,
                    "score": score,
                    "metrics": metrics_summary or {},
                },
            )
        )

    def log_info(self, message: str) -> None:
        self.entries.append(
            TraceEntry(
                timestamp=time.time(),
                phase=self._current_phase,
                event="info",
                detail={"message": _sanitise_text(message)},
            )
        )

    def get_traces(self) -> list[dict[str, Any]]:
        """Return all entries as serialisable dicts."""
        return [
            {
                "timestamp": e.timestamp,
                "phase": e.phase,
                "event": e.event,
                "detail": e.detail,
            }
            for e in self.entries
        ]
