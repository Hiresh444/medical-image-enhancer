"""Async pipeline runner — wraps ``pipeline.runner.run_pipeline()`` in a thread
so the API can return ``run_id`` immediately and let the frontend poll for status.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from backend.config import OUTPUT_DIR, OPENAI_MODEL, MAX_ITERS
from pipeline.storage import (
    generate_run_id,
    init_db,
    insert_pending_run,
    update_run_status,
    get_run,
)

logger = logging.getLogger(__name__)


def start_run(
    file_path: str,
    *,
    genai: bool = False,
    model: str | None = None,
    max_iters: int | None = None,
    output_dir: str | None = None,
) -> str:
    """Launch a pipeline run in a background thread.

    Returns the ``run_id`` immediately so the caller can poll for status.
    """
    init_db()

    run_id = generate_run_id()
    input_filename = os.path.basename(file_path)
    insert_pending_run(run_id, input_filename)

    effective_model = model or OPENAI_MODEL
    effective_iters = max_iters if max_iters is not None else MAX_ITERS
    effective_output = output_dir or OUTPUT_DIR

    thread = threading.Thread(
        target=_run_in_thread,
        args=(run_id, file_path, effective_output, genai, effective_model, effective_iters),
        daemon=True,
    )
    thread.start()
    return run_id


def _run_in_thread(
    run_id: str,
    file_path: str,
    output_dir: str,
    genai: bool,
    model: str,
    max_iters: int,
) -> None:
    """Execute the pipeline synchronously inside a thread."""
    from pipeline.runner import run_pipeline
    import pipeline.runner as _runner_mod

    try:
        update_run_status(run_id, "running")
        logger.info("Pipeline run %s started (genai=%s, model=%s)", run_id, genai, model)

        # run_pipeline imports generate_run_id via
        #   ``from pipeline.storage import generate_run_id``
        # which binds a local name in the runner module.  We must patch
        # THAT reference so the pipeline uses our predetermined run_id.
        _orig_gen = _runner_mod.generate_run_id
        _runner_mod.generate_run_id = lambda: run_id  # deterministic for this run

        try:
            ctx = run_pipeline(
                input_path=file_path,
                output_dir=output_dir,
                genai=genai,
                model=model,
                max_iters=max_iters,
                save_artifacts=True,
                no_show=True,
            )
        finally:
            _runner_mod.generate_run_id = _orig_gen  # restore

        # run_pipeline already persisted via _persist_run; ensure status correct
        status = "completed"
        if ctx.get("genai_fell_back"):
            status = "completed"  # still a valid result
        update_run_status(run_id, status)

        logger.info("Pipeline run %s completed.", run_id)

    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        update_run_status(run_id, "error")


def get_run_status(run_id: str) -> str:
    """Read the current status of a run from the database."""
    data = get_run(run_id)
    if data is None:
        return "not_found"
    return data.get("status", "unknown")
