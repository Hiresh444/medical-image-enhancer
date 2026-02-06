"""GenAI orchestration layer using the OpenAI Agents SDK.

Three LLM-backed agents wrap the deterministic enhancement pipeline:
  1. **GenAIPlannerAgent** – generates a structured enhancement plan (JSON)
  2. **GenAITuningAgent**  – iteratively refines parameters via tool calls
  3. **GenAIExplainabilityAgent** – produces a clinician-friendly summary

**Privacy guarantee**: Only numeric metrics, issue labels, and non-PHI DICOM
metadata are ever included in prompts.  Pixel arrays stay in an in-memory
store and are referenced by opaque string IDs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from agents import Agent, ModelSettings, Runner
from agents.exceptions import AgentsException

from schemas import (
    EnhancementPlan,
    ExplainabilityReport,
    GenAIContext,
    IterationRecord,
    PARAM_BOUNDS,
)
from tools import (
    clear_image_store,
    register_image,
    tool_apply_enhancement,
    tool_get_metrics,
    tool_score_plan,
    tool_validate,
    _clamp_params,
)
from utils import (
    apply_enhancements_from_params,
    compute_metrics,
    compute_validation,
    THRESHOLDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_LLM_CALLS = 10  # hard-cap across the entire pipeline
_FALLBACK_MODEL = "o4-mini"

# ---------------------------------------------------------------------------
# Metadata sanitisation (defence-in-depth against prompt injection)
# ---------------------------------------------------------------------------

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitise_metadata(metadata: dict[str, str], max_len: int = 100) -> dict[str, str]:
    """Strip control characters and truncate values.  Metadata fields are
    treated as *untrusted* text that must never override system instructions."""
    safe: dict[str, str] = {}
    allowed_keys = {"Modality", "BodyPartExamined", "StudyDescription"}
    for k, v in metadata.items():
        if k not in allowed_keys:
            continue
        v = _CTRL_RE.sub("", str(v))[:max_len]
        safe[k] = v
    return safe


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are GenAIPlannerAgent, an expert in medical imaging quality assurance.

Your task: given detected quality issues and numeric metrics for a DICOM image,
produce a single JSON enhancement plan that a deterministic pipeline will execute.

## CONSTRAINTS
- Preserve anatomical structures — no aggressive processing.
- Avoid halos from over-sharpening.
- Conservative enhancement preferred over aggressive.
- CPU-only execution; plan must be computationally lightweight.
- NEVER request operations not in the valid set.

## VALID OPERATIONS (in pipeline order)
1. "denoise"       — wavelet denoising (pre-enhancement cleanup)
2. "clahe"         — contrast-limited adaptive histogram equalisation
3. "gamma"         — gamma correction for shadow/highlight adjustment
4. "unsharp"       — unsharp mask for sharpening
5. "post_denoise"  — light wavelet denoise after sharpening

## PARAMETER BOUNDS
{param_bounds}

## THRESHOLDS (for reference)
{thresholds}

## INPUT
<metadata>
{metadata}
</metadata>

Detected issues: {issues}
Current metrics: {metrics}

## OUTPUT
Respond ONLY with a valid EnhancementPlan JSON object.  If no enhancement is
needed, set "stop_reason" to a short explanation and leave "recommended_ops"
empty.
"""

TUNING_SYSTEM_PROMPT = """\
You are GenAITuningAgent, an expert at iteratively tuning medical image
enhancement parameters to maximise quality while preserving anatomy.

## OBJECTIVE
Maximise the objective score (higher is better):
  score = 0.35 * contrast_gain + 0.35 * sharpness_gain
          - 0.30 * noise_penalty - 5.0 * niqe_degradation
          - 10.0 * (0 if passes else 1)

## WORKFLOW (repeat up to {max_iters} iterations)
1. Construct an EnhancementPlan JSON with your chosen parameters.
2. Call tool_apply_enhancement with the plan JSON to run the pipeline.
3. Call tool_validate with the original and enhanced image IDs.
4. Call tool_score_plan with the validation JSON to get the objective score.
5. If the score is satisfactory or you have exhausted iterations, respond
   with your BEST plan as the final output.
6. Otherwise, adjust parameters and repeat from step 1.

## PARAMETER BOUNDS
{param_bounds}

## SEED PLAN (from PlannerAgent)
{seed_plan}

## BASELINE
Original image ID: {original_id}
Original metrics: {metrics}

## RULES
- Try 2–3 meaningfully different parameter sets.
- Do NOT repeat the same parameters.
- Prefer plans that PASS validation.
- When in doubt, be MORE conservative (smaller clip_limit, lower unsharp_amount).

Respond with your FINAL best EnhancementPlan JSON as your last message.
"""

EXPLAINABILITY_SYSTEM_PROMPT = """\
You are GenAIExplainabilityAgent.  You write concise, clinician-friendly
explanations of medical image quality assessment results.

Write EXACTLY five sections (one short paragraph each, 2–3 sentences max):
1. **Detected Issues** – what quality problems were found and their severity.
2. **Corrective Measures** – what was recommended and the clinical rationale.
3. **Enhancement Applied** – which operations ran and their parameter highlights.
4. **Validation Outcome** – SSIM/PSNR/quality-improvement results and meaning.
5. **Limitations** – safe-use warning; state this is NOT for clinical diagnosis.

Total length: 10–14 lines.  Use plain language a radiologist can scan quickly.
Do NOT include raw JSON, code, or pixel values.
"""


# ---------------------------------------------------------------------------
# Agent factories (created per-run to inject dynamic context)
# ---------------------------------------------------------------------------

def _fmt_param_bounds() -> str:
    lines = []
    for k, (lo, hi) in PARAM_BOUNDS.items():
        lines.append(f"  {k}: [{lo}, {hi}]")
    return "\n".join(lines)


def _fmt_thresholds() -> str:
    return "\n".join(f"  {k}: {v}" for k, v in THRESHOLDS.items())


def _build_planner_agent(
    ctx: GenAIContext,
    metadata: dict[str, str],
    model: str,
) -> Agent:
    instructions = PLANNER_SYSTEM_PROMPT.format(
        param_bounds=_fmt_param_bounds(),
        thresholds=_fmt_thresholds(),
        metadata=json.dumps(metadata, indent=2),
        issues=json.dumps(ctx.issues),
        metrics=json.dumps(ctx.metrics, indent=2),
    )
    return Agent(
        name="GenAIPlannerAgent",
        model=model,
        instructions=instructions,
        output_type=EnhancementPlan,
        model_settings=ModelSettings(temperature=0.2),
        tools=[tool_get_metrics],
    )


def _build_tuning_agent(
    seed_plan: EnhancementPlan,
    original_id: str,
    metrics: dict[str, float],
    model: str,
    max_iters: int,
) -> Agent:
    instructions = TUNING_SYSTEM_PROMPT.format(
        max_iters=max_iters,
        param_bounds=_fmt_param_bounds(),
        seed_plan=seed_plan.model_dump_json(indent=2),
        original_id=original_id,
        metrics=json.dumps(metrics, indent=2),
    )
    return Agent(
        name="GenAITuningAgent",
        model=model,
        instructions=instructions,
        output_type=EnhancementPlan,
        model_settings=ModelSettings(temperature=0.3),
        tools=[tool_apply_enhancement, tool_validate, tool_score_plan],
    )


def _build_explainability_agent(model: str) -> Agent:
    return Agent(
        name="GenAIExplainabilityAgent",
        model=model,
        instructions=EXPLAINABILITY_SYSTEM_PROMPT,
        output_type=ExplainabilityReport,
        model_settings=ModelSettings(temperature=0.4),
    )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_agent_with_fallback(
    agent: Agent,
    input_text: str,
    fallback_model: str = _FALLBACK_MODEL,
    max_turns: int = 25,
) -> Any:
    """Run agent synchronously.  On failure, retry once with *fallback_model*."""
    try:
        result = Runner.run_sync(agent, input=input_text, max_turns=max_turns)
        return result.final_output
    except AgentsException as exc:
        logger.warning(
            "Agent '%s' failed with %s (%s); retrying with %s",
            agent.name, type(exc).__name__, exc, fallback_model,
        )
        fallback_agent = agent.clone(model=fallback_model)
        result = Runner.run_sync(fallback_agent, input=input_text, max_turns=max_turns)
        return result.final_output


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class GenAIPipelineResult:
    """Everything produced by the GenAI pipeline, ready for report generation."""

    plan: EnhancementPlan | None = None
    iterations: list[IterationRecord] = field(default_factory=list)
    best_plan: EnhancementPlan | None = None
    enhanced_image: np.ndarray | None = None
    applied_ops: list[str] = field(default_factory=list)
    enhanced_metrics: dict[str, float] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    explainability: ExplainabilityReport | None = None
    model_name: str = ""
    prompts_used: list[str] = field(default_factory=list)
    llm_call_count: int = 0
    fell_back_to_deterministic: bool = False
    plan_only: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_genai_pipeline(
    image: np.ndarray,
    metadata: dict[str, str],
    issues: list[str],
    metrics: dict[str, float],
    *,
    model: str = "gpt-4.1-mini",
    max_iters: int = 4,
    plan_only: bool = False,
) -> GenAIPipelineResult:
    """Run the full GenAI-augmented enhancement pipeline.

    Parameters
    ----------
    image : np.ndarray
        Normalised [0, 1] grayscale image.
    metadata : dict
        Non-PHI DICOM metadata (Modality, BodyPartExamined, StudyDescription).
    issues : list[str]
        Issue codes from deterministic detection.
    metrics : dict
        Quality metrics from deterministic detection.
    model : str
        OpenAI model name for primary agents.
    max_iters : int
        Maximum tuning iterations.
    plan_only : bool
        If True, return the plan JSON without executing enhancement.

    Returns
    -------
    GenAIPipelineResult
        Contains plan, iterations, enhanced image, validation, explainability.
    """
    result = GenAIPipelineResult(model_name=model)
    safe_metadata = _sanitise_metadata(metadata)

    # Register the original image ------------------------------------------
    original_id = register_image(image, "original")

    ctx = GenAIContext(
        metrics=metrics,
        issues=issues,
        thresholds=dict(THRESHOLDS),
        metadata=safe_metadata,
        image_id=original_id,
    )

    # ------------------------------------------------------------------ #
    # Phase 1: Planner                                                    #
    # ------------------------------------------------------------------ #
    try:
        planner = _build_planner_agent(ctx, safe_metadata, model)
        result.prompts_used.append("GenAIPlannerAgent system prompt (issues + metrics)")

        plan: EnhancementPlan = _run_agent_with_fallback(
            planner,
            "Generate an enhancement plan for the image based on the detected issues and metrics.",
            max_turns=5,
        )
        result.plan = plan
        result.llm_call_count += 1
        logger.info("Planner produced plan: %s", plan.model_dump_json(indent=2))

        # Early exit: no enhancement needed
        if plan.stop_reason:
            result.best_plan = plan
            result.enhanced_image = image.copy()
            result.enhanced_metrics = metrics.copy()
            result.plan_only = True
            clear_image_store()
            return result

    except Exception as exc:
        logger.error("Planner failed: %s — falling back to deterministic", exc)
        result.error = f"Planner failed: {exc}"
        result.fell_back_to_deterministic = True
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 1.5: Plan-only mode                                           #
    # ------------------------------------------------------------------ #
    if plan_only:
        result.best_plan = plan
        result.plan_only = True
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 2: Tuning loop                                                #
    # ------------------------------------------------------------------ #
    best_plan = plan

    try:
        tuner = _build_tuning_agent(plan, original_id, metrics, model, max_iters)
        result.prompts_used.append("GenAITuningAgent system prompt (seed plan + objective)")

        tuned_plan: EnhancementPlan = _run_agent_with_fallback(
            tuner,
            (
                f"Original image ID: {original_id}\n"
                f"Seed plan:\n{plan.model_dump_json(indent=2)}\n"
                f"Baseline metrics:\n{json.dumps(metrics, indent=2)}\n\n"
                f"Run up to {max_iters} iterations.  Use the tools to test "
                f"each plan variant and pick the best one."
            ),
            max_turns=max_iters * 4 + 2,
        )
        result.llm_call_count += 1  # count the tuning run as 1 orchestrated call

        # The tuning agent returns its best plan as structured output
        best_plan = _clamp_params(tuned_plan)

    except Exception as exc:
        logger.warning("Tuning failed: %s — using planner's seed plan", exc)
        result.prompts_used.append(f"Tuning fallback: {exc}")

    result.best_plan = best_plan

    # ------------------------------------------------------------------ #
    # Phase 3: Execute best plan deterministically                        #
    # ------------------------------------------------------------------ #
    try:
        enhanced, applied_ops = apply_enhancements_from_params(image, best_plan)
        result.enhanced_image = enhanced
        result.applied_ops = applied_ops
        result.enhanced_metrics = compute_metrics(enhanced)
    except Exception as exc:
        logger.error("Enhancement execution failed: %s", exc)
        result.error = f"Enhancement execution failed: {exc}"
        result.enhanced_image = image.copy()
        result.enhanced_metrics = metrics.copy()
        result.fell_back_to_deterministic = True
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 4: Validate                                                   #
    # ------------------------------------------------------------------ #
    try:
        validation = compute_validation(image, enhanced)
        result.validation = validation
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        result.validation = {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Phase 5: Explainability                                             #
    # ------------------------------------------------------------------ #
    try:
        explainer = _build_explainability_agent(model)
        result.prompts_used.append("GenAIExplainabilityAgent system prompt")

        explainability_input = (
            f"Issues detected: {json.dumps(issues)}\n"
            f"Applied operations: {json.dumps(applied_ops)}\n"
            f"SSIM: {validation.get('ssim', 'N/A')}\n"
            f"PSNR: {validation.get('psnr', 'N/A')}\n"
            f"Quality improvement: {validation.get('quality_improvement', 'N/A')}\n"
            f"Passes: {validation.get('passes', 'N/A')}\n"
            f"NIQE before: {validation.get('niqe_before', 'N/A')}\n"
            f"NIQE after: {validation.get('niqe_after', 'N/A')}\n"
            f"Parameters used: {best_plan.params.model_dump_json()}\n"
            f"Risk warnings: {json.dumps(best_plan.risk_warnings)}\n"
        )

        report: ExplainabilityReport = _run_agent_with_fallback(
            explainer,
            explainability_input,
            max_turns=3,
        )
        result.explainability = report
        result.llm_call_count += 1

    except Exception as exc:
        logger.warning("Explainability agent failed: %s", exc)
        result.explainability = ExplainabilityReport(
            detected_issues=f"Issues detected: {', '.join(issues) or 'none'}.",
            corrective_measures="Standard deterministic recommendations applied.",
            enhancement_applied=f"Applied: {', '.join(applied_ops) or 'none'}.",
            validation_outcome="See validation metrics table in report.",
            limitations=(
                "This tool is for quality assurance research only and is "
                "NOT intended for clinical diagnosis or patient care decisions."
            ),
        )

    clear_image_store()
    return result
