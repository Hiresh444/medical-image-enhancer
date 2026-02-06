"""Unified pipeline runner — single entry point for CLI and Flask.

``run_pipeline()`` encapsulates the full flow:
  load → detect → enhance → validate → report → save artifacts → persist to DB.

Both ``main.py`` (CLI) and ``app.py`` (Flask) call this function.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from pipeline.agent_logger import AgentTraceLogger
from pipeline.core_agents import (
    QualityDetectionAgent,
    RecommendationAgent,
    EnhancementAgent,
    ValidationAgent,
    ReportAgent,
)
from pipeline.dicom_io import load_dicom, normalize_image, save_visuals
from pipeline.metrics import compute_metrics
from pipeline.storage import generate_run_id, save_run, init_db

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_dir: str = "outputs",
    *,
    genai: bool = False,
    model: str | None = None,
    max_iters: int = 4,
    plan_only: bool = False,
    save_artifacts: bool = True,
    no_show: bool = True,
) -> dict[str, Any]:
    """Run the full medical imaging QA pipeline.

    Parameters
    ----------
    input_path : str
        Path to a DICOM file.
    output_dir : str
        Directory for output artefacts.
    genai : bool
        Enable GenAI agentic mode.
    model : str | None
        OpenAI model name override.
    max_iters : int
        Max GenAI tuning iterations.
    plan_only : bool
        Return plan JSON without executing enhancement (GenAI only).
    save_artifacts : bool
        Save report + images to disk and DB.
    no_show : bool
        Suppress matplotlib display.

    Returns
    -------
    dict
        Full pipeline context including run_id, metrics, validation, report, etc.
    """
    # Ensure DB is initialised
    init_db()

    run_id = generate_run_id()
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # ------------------------------------------------------------------ #
    # Load & normalise                                                    #
    # ------------------------------------------------------------------ #
    image_raw, metadata = load_dicom(input_path)
    image = normalize_image(image_raw)

    # ------------------------------------------------------------------ #
    # Detection                                                           #
    # ------------------------------------------------------------------ #
    detector = QualityDetectionAgent()
    detection = detector.run(image)

    # ------------------------------------------------------------------ #
    # Branch: GenAI vs deterministic                                      #
    # ------------------------------------------------------------------ #
    if genai:
        context = _run_genai_path(
            run_id=run_id,
            image=image,
            metadata=metadata,
            detection=detection,
            model=model,
            max_iters=max_iters,
            plan_only=plan_only,
            input_path=input_path,
            output_dir=output_dir,
            base_name=base_name,
            save_artifacts=save_artifacts,
        )
    else:
        context = _run_deterministic_path(
            run_id=run_id,
            image=image,
            metadata=metadata,
            detection=detection,
            input_path=input_path,
            output_dir=output_dir,
            base_name=base_name,
            save_artifacts=save_artifacts,
        )

    return context


# ====================================================================== #
# Deterministic path                                                      #
# ====================================================================== #


def _run_deterministic_path(
    *,
    run_id: str,
    image: np.ndarray,
    metadata: dict,
    detection,
    input_path: str,
    output_dir: str,
    base_name: str,
    save_artifacts: bool,
) -> dict[str, Any]:
    recommender = RecommendationAgent()
    enhancer = EnhancementAgent()
    validator = ValidationAgent()
    reporter = ReportAgent()

    recommendations = recommender.run(detection)

    if detection.issues:
        enhancement = enhancer.run(image, recommendations)
        enhanced_image = enhancement.image
        applied_ops = enhancement.applied_ops
        enhanced_metrics = enhancement.metrics
    else:
        enhanced_image = image
        applied_ops = []
        enhanced_metrics = detection.metrics

    validation = validator.run(image, enhanced_image, detection)

    visuals: dict[str, str] = {}
    report_path = ""
    before_after_path = ""

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        visuals = save_visuals(image, enhanced_image, output_dir, base_name)
        before_after_path = visuals.get("before_after", "")

    context: dict[str, Any] = {
        "run_id": run_id,
        "input_path": input_path,
        "metadata": metadata,
        "issues": detection.issues,
        "recommendations": recommendations.recommendations,
        "applied_ops": applied_ops,
        "metrics_before": detection.metrics,
        "metrics_after": enhanced_metrics,
        "validation": validation,
        "visuals": visuals,
        "notes": validation.notes,
        "enhanced_image": enhanced_image,
        "original_image": image,
    }

    report_md = reporter.run(context)
    context["report_md"] = report_md

    if save_artifacts:
        report_path = os.path.join(output_dir, f"{base_name}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        context["report_path"] = report_path

        # Persist to DB
        _persist_run(
            run_id=run_id,
            input_filename=os.path.basename(input_path),
            metadata=metadata,
            issues=detection.issues,
            metrics_before=detection.metrics,
            metrics_after=enhanced_metrics,
            plan_json="",
            validation=validation,
            applied_ops=applied_ops,
            explainability={},
            report_path=report_path,
            before_after_path=before_after_path,
            agent_logs=[],
            status=validation.status,
        )

    return context


# ====================================================================== #
# GenAI path                                                              #
# ====================================================================== #


def _run_genai_path(
    *,
    run_id: str,
    image: np.ndarray,
    metadata: dict,
    detection,
    model: str | None,
    max_iters: int,
    plan_only: bool,
    input_path: str,
    output_dir: str,
    base_name: str,
    save_artifacts: bool,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GenAI mode requires the OPENAI_API_KEY environment variable."
        )

    from pipeline.genai_agents import run_genai_pipeline

    trace_logger = AgentTraceLogger()

    logger.info(
        "Running GenAI pipeline (model=%s, max_iters=%d)", model, max_iters
    )

    result = run_genai_pipeline(
        image=image,
        metadata=metadata,
        issues=detection.issues,
        metrics=detection.metrics,
        model=model,
        max_iters=max_iters,
        plan_only=plan_only,
        trace_logger=trace_logger,
    )

    # Fallback to deterministic if GenAI failed
    if result.fell_back_to_deterministic:
        logger.warning(
            "GenAI pipeline failed (%s) — falling back to deterministic path.",
            result.error,
        )
        ctx = _run_deterministic_path(
            run_id=run_id,
            image=image,
            metadata=metadata,
            detection=detection,
            input_path=input_path,
            output_dir=output_dir,
            base_name=base_name,
            save_artifacts=save_artifacts,
        )
        ctx["genai_error"] = result.error
        ctx["genai_fell_back"] = True
        return ctx

    # Plan-only mode
    if result.plan_only and result.plan:
        return {
            "run_id": run_id,
            "plan_only": True,
            "plan": result.plan,
            "stop_reason": result.plan.stop_reason,
        }

    # Full GenAI execution
    enhanced_image = (
        result.enhanced_image if result.enhanced_image is not None else image
    )
    enhanced_metrics = result.enhanced_metrics or detection.metrics

    validator = ValidationAgent()
    validation = validator.run(image, enhanced_image, detection)

    visuals: dict[str, str] = {}
    report_path = ""
    before_after_path = ""

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        visuals = save_visuals(image, enhanced_image, output_dir, base_name)
        before_after_path = visuals.get("before_after", "")

    # Recommendations from GenAI plan
    recommendations_text: list[str] = []
    if result.best_plan:
        recommendations_text.append(result.best_plan.rationale)
        for w in result.best_plan.risk_warnings:
            recommendations_text.append(f"⚠️ {w}")
    if not recommendations_text:
        recommender = RecommendationAgent()
        rec_result = recommender.run(detection)
        recommendations_text = rec_result.recommendations

    context: dict[str, Any] = {
        "run_id": run_id,
        "input_path": input_path,
        "metadata": metadata,
        "issues": detection.issues,
        "recommendations": recommendations_text,
        "applied_ops": result.applied_ops,
        "metrics_before": detection.metrics,
        "metrics_after": enhanced_metrics,
        "validation": validation,
        "visuals": visuals,
        "notes": validation.notes,
        "enhanced_image": enhanced_image,
        "original_image": image,
        # GenAI-specific
        "genai_plan": result.best_plan,
        "genai_iterations": result.iterations,
        "genai_model": result.model_name,
        "genai_max_iters": max_iters,
        "genai_llm_calls": result.llm_call_count,
        "genai_prompts": result.prompts_used,
        "genai_explainability": result.explainability,
        "agent_traces": result.agent_traces,
    }

    reporter = ReportAgent()
    report_md = reporter.run(context)
    context["report_md"] = report_md

    if save_artifacts:
        report_path = os.path.join(output_dir, f"{base_name}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        context["report_path"] = report_path

        # Build explainability dict for storage
        expl_dict: dict[str, Any] = {}
        if result.explainability:
            try:
                expl_dict = result.explainability.model_dump()
            except Exception:
                expl_dict = {"text": str(result.explainability)}

        plan_json_str = ""
        if result.best_plan:
            plan_json_str = result.best_plan.model_dump_json(indent=2)

        # Build validation dict for storage
        val_dict: dict[str, Any] = {}
        if hasattr(validation, "__dict__"):
            for k, v in validation.__dict__.items():
                if not k.startswith("_"):
                    val_dict[k] = v
        elif isinstance(validation, dict):
            val_dict = validation

        _persist_run(
            run_id=run_id,
            input_filename=os.path.basename(input_path),
            metadata=metadata,
            issues=detection.issues,
            metrics_before=detection.metrics,
            metrics_after=enhanced_metrics,
            plan_json=plan_json_str,
            validation=val_dict if val_dict else validation,
            applied_ops=result.applied_ops,
            explainability=expl_dict,
            report_path=report_path,
            before_after_path=before_after_path,
            agent_logs=result.agent_traces,
            status=validation.status if hasattr(validation, "status") else "completed",
            genai_model=result.model_name,
            genai_llm_calls=result.llm_call_count,
        )

    return context


# ---------------------------------------------------------------------------
# DB persistence helper
# ---------------------------------------------------------------------------


def _persist_run(
    *,
    run_id: str,
    input_filename: str,
    metadata: dict,
    issues: list,
    metrics_before: dict,
    metrics_after: dict,
    plan_json: str,
    validation,
    applied_ops: list,
    explainability: dict | str,
    report_path: str,
    before_after_path: str,
    agent_logs: list,
    status: str = "completed",
    genai_model: str = "",
    genai_llm_calls: int = 0,
) -> None:
    """Persist a run to SQLite, converting ValidationResult if needed."""
    val_dict: dict[str, Any] = {}
    if hasattr(validation, "__dict__"):
        for k, v in validation.__dict__.items():
            if not k.startswith("_"):
                val_dict[k] = v
    elif isinstance(validation, dict):
        val_dict = validation

    try:
        save_run(
            run_id=run_id,
            input_filename=input_filename,
            metadata_summary=metadata,
            issues=issues,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            plan_json=plan_json,
            validation=val_dict,
            applied_ops=applied_ops,
            explainability=explainability if isinstance(explainability, dict) else {"text": str(explainability)},
            report_path=report_path,
            before_after_path=before_after_path,
            agent_logs=agent_logs,
            status=status,
            genai_model=genai_model,
            genai_llm_calls=genai_llm_calls,
        )
        logger.info("Run %s persisted to DB.", run_id)
    except Exception as exc:
        logger.error("Failed to persist run %s: %s", run_id, exc)
