"""GenAI orchestration layer using the OpenAI Agents SDK.

Three LLM-backed agents wrap the deterministic enhancement pipeline:
  1. **GenAIPlannerAgent** — generates a structured enhancement plan (JSON)
  2. **GenAITuningAgent**  — iteratively refines parameters via tool calls
  3. **GenAIExplainabilityAgent** — produces a clinician-friendly summary

**Privacy guarantee**: Only numeric metrics, issue labels, and non-PHI DICOM
metadata are ever included in prompts.  Pixel arrays stay in an in-memory
store and are referenced by opaque string IDs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from agents import Agent, ModelSettings, Runner
from agents.exceptions import AgentsException

from pipeline.schemas import (
    EnhancementPlan,
    ExplainabilityReport,
    GenAIContext,
    IterationRecord,
    PARAM_BOUNDS,
)
from pipeline.tools import (
    clear_image_store,
    register_image,
    tool_apply_enhancement,
    tool_get_metrics,
    tool_score_plan,
    tool_validate,
    _clamp_params,
)
from pipeline.metrics import (
    compute_metrics,
    compute_validation,
    THRESHOLDS,
)
from pipeline.enhancement import apply_enhancements_from_params
from pipeline.agent_logger import AgentTraceLogger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

_MAX_LLM_CALLS = int(os.environ.get("MDIMG_MAX_LLM_CALLS", "10"))
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
_DEFAULT_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
_DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))
_FALLBACK_MODEL = "o4-mini"

# Models that reject the 'temperature' parameter (reasoning / mini models)
_NO_TEMPERATURE_PATTERNS = ("o1", "o3", "o4", "gpt-5")


def _safe_model_settings(
    model: str,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> ModelSettings:
    """Build ModelSettings, omitting *temperature* for models that reject it."""
    model_lower = model.lower()
    if any(model_lower.startswith(p) for p in _NO_TEMPERATURE_PATTERNS):
        return ModelSettings(max_tokens=max_tokens)
    return ModelSettings(temperature=temperature, max_tokens=max_tokens)

# ---------------------------------------------------------------------------
# Metadata sanitisation (defence-in-depth against prompt injection)
# ---------------------------------------------------------------------------

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitise_metadata(metadata: dict[str, str], max_len: int = 100) -> dict[str, str]:
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
6. "bilateral"     — bilateral filter for edge-preserving denoise (optional)
7. "tv_denoise"    — total-variation denoise (optional)

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
          - halo_penalty - entropy_penalty
          + snr_reward + histogram_spread_reward

## SAFEGUARDS (automatic — applied by the pipeline)
- Halo detection: edge_ratio > 1.5 triggers unsharp reduction
- Noise amplification guard: sigma_after > 1.3 * sigma_before triggers auto-denoise
- Over-processing guard: NIQE degradation > 0.5 triggers blend-back

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
- Consider bilateral or tv_denoise for edge-preserving smoothing if noise is high.

Respond with your FINAL best EnhancementPlan JSON as your last message.
"""

EXPLAINABILITY_SYSTEM_PROMPT = """\
You are GenAIExplainabilityAgent.  You write concise, clinician-friendly
explanations of medical image quality assessment results.

Write EXACTLY eight fields in your response:

1. **detected_issues** – what quality problems were found and their severity (2-3 sentences).
2. **corrective_measures** – what was recommended and the clinical rationale (2-3 sentences).
3. **enhancement_applied** – which operations ran and their parameter highlights (2-3 sentences).
4. **validation_outcome** – SSIM/PSNR/quality-improvement results and meaning (2-3 sentences).
5. **limitations** – safe-use warning; state this is NOT for clinical diagnosis (2-3 sentences).
6. **image_summary** – non-PHI summary: modality, body part if available, issues detected, why actions were suggested, expected tradeoffs (2-3 sentences).
7. **actionable_suggestions** – list of 2-4 actionable suggestions (e.g., "if still low contrast, consider increasing CLAHE clip_limit to 0.03").
8. **next_steps** – list of 2-3 recommended next steps for the user.

Do NOT include raw JSON, code, pixel values, or PHI.
Use plain language a radiologist can scan quickly.
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
        model_settings=_safe_model_settings(model, _DEFAULT_TEMPERATURE),
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
        model_settings=_safe_model_settings(model, min(_DEFAULT_TEMPERATURE + 0.1, 0.4)),
        tools=[tool_apply_enhancement, tool_validate, tool_score_plan],
    )


def _build_explainability_agent(model: str) -> Agent:
    return Agent(
        name="GenAIExplainabilityAgent",
        model=model,
        instructions=EXPLAINABILITY_SYSTEM_PROMPT,
        output_type=ExplainabilityReport,
        model_settings=_safe_model_settings(model, min(_DEFAULT_TEMPERATURE + 0.2, 0.5)),
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
    agent_traces: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_genai_pipeline(
    image: np.ndarray,
    metadata: dict[str, str],
    issues: list[str],
    metrics: dict[str, float],
    *,
    model: str | None = None,
    max_iters: int = 4,
    plan_only: bool = False,
    trace_logger: AgentTraceLogger | None = None,
) -> GenAIPipelineResult:
    """Run the full GenAI-augmented enhancement pipeline.

    Parameters
    ----------
    image : np.ndarray
        Normalised [0, 1] grayscale image.
    metadata : dict
        Non-PHI DICOM metadata.
    issues : list[str]
        Issue codes from deterministic detection.
    metrics : dict
        Quality metrics from deterministic detection.
    model : str | None
        OpenAI model name.  ``None`` → ``OPENAI_MODEL`` env var → ``gpt-5-mini``.
    max_iters : int
        Maximum tuning iterations.
    plan_only : bool
        If True, return the plan JSON without executing enhancement.
    trace_logger : AgentTraceLogger | None
        Optional logger for agent traces.

    Returns
    -------
    GenAIPipelineResult
    """
    if model is None:
        model = _DEFAULT_MODEL

    result = GenAIPipelineResult(model_name=model)
    safe_metadata = _sanitise_metadata(metadata)
    tl = trace_logger or AgentTraceLogger()

    # Cost guard counter
    llm_calls_used = 0

    def _guard() -> bool:
        nonlocal llm_calls_used
        if llm_calls_used >= _MAX_LLM_CALLS:
            logger.warning("Cost guard: reached %d LLM calls — stopping.", _MAX_LLM_CALLS)
            tl.log_info(f"Cost guard triggered at {llm_calls_used} calls.")
            return True
        return False

    # Register the original image
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
        tl.log_phase_start("planner")
        tl.log_prompt("System prompt with issues + metrics + param bounds")

        planner = _build_planner_agent(ctx, safe_metadata, model)
        result.prompts_used.append("GenAIPlannerAgent system prompt (issues + metrics)")

        plan: EnhancementPlan = _run_agent_with_fallback(
            planner,
            "Generate an enhancement plan for the image based on the detected issues and metrics.",
            max_turns=5,
        )
        result.plan = plan
        llm_calls_used += 1
        result.llm_call_count += 1
        logger.info("Planner produced plan: %s", plan.model_dump_json(indent=2))
        tl.log_phase_end("planner", f"Plan: {len(plan.recommended_ops)} ops")

        if plan.stop_reason:
            result.best_plan = plan
            result.enhanced_image = image.copy()
            result.enhanced_metrics = metrics.copy()
            result.plan_only = True
            result.agent_traces = tl.get_traces()
            clear_image_store()
            return result

    except Exception as exc:
        logger.error("Planner failed: %s — falling back to deterministic", exc)
        tl.log_phase_end("planner", f"FAILED: {exc}")
        result.error = f"Planner failed: {exc}"
        result.fell_back_to_deterministic = True
        result.agent_traces = tl.get_traces()
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 1.5: Plan-only mode                                           #
    # ------------------------------------------------------------------ #
    if plan_only:
        result.best_plan = plan
        result.plan_only = True
        result.agent_traces = tl.get_traces()
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 2: Tuning loop                                                #
    # ------------------------------------------------------------------ #
    best_plan = plan

    if not _guard():
        try:
            tl.log_phase_start("tuning")
            tl.log_prompt("System prompt with seed plan + objective + param bounds")

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
            llm_calls_used += 1
            result.llm_call_count += 1

            best_plan = _clamp_params(tuned_plan)
            tl.log_phase_end("tuning", "Tuning complete — best plan selected")

        except Exception as exc:
            logger.warning("Tuning failed: %s — using planner's seed plan", exc)
            tl.log_phase_end("tuning", f"FAILED: {exc} — using seed plan")
            result.prompts_used.append(f"Tuning fallback: {exc}")

    result.best_plan = best_plan

    # ------------------------------------------------------------------ #
    # Phase 3: Execute best plan deterministically                        #
    # ------------------------------------------------------------------ #
    try:
        tl.log_phase_start("execution")
        enhanced, applied_ops = apply_enhancements_from_params(image, best_plan)
        result.enhanced_image = enhanced
        result.applied_ops = applied_ops
        result.enhanced_metrics = compute_metrics(enhanced)
        tl.log_phase_end("execution", f"Applied {len(applied_ops)} ops")
    except Exception as exc:
        logger.error("Enhancement execution failed: %s", exc)
        tl.log_phase_end("execution", f"FAILED: {exc}")
        result.error = f"Enhancement execution failed: {exc}"
        result.enhanced_image = image.copy()
        result.enhanced_metrics = metrics.copy()
        result.fell_back_to_deterministic = True
        result.agent_traces = tl.get_traces()
        clear_image_store()
        return result

    # ------------------------------------------------------------------ #
    # Phase 4: Validate                                                   #
    # ------------------------------------------------------------------ #
    try:
        tl.log_phase_start("validation")
        validation = compute_validation(image, enhanced)
        result.validation = validation
        tl.log_phase_end(
            "validation",
            f"SSIM={validation.get('ssim', 0):.3f} PSNR={validation.get('psnr', 0):.1f}",
        )
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        tl.log_phase_end("validation", f"FAILED: {exc}")
        result.validation = {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Phase 5: Explainability                                             #
    # ------------------------------------------------------------------ #
    if not _guard():
        try:
            tl.log_phase_start("explainability")
            tl.log_prompt("System prompt for clinician-friendly explanation")

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
                f"SNR before: {validation.get('snr_before', 'N/A')}\n"
                f"SNR after: {validation.get('snr_after', 'N/A')}\n"
                f"CNR before: {validation.get('cnr_before', 'N/A')}\n"
                f"CNR after: {validation.get('cnr_after', 'N/A')}\n"
                f"Edge ratio: {validation.get('edge_ratio', 'N/A')}\n"
                f"Entropy change: {validation.get('entropy_change', 'N/A')}\n"
                f"Parameters used: {best_plan.params.model_dump_json()}\n"
                f"Risk warnings: {json.dumps(best_plan.risk_warnings)}\n"
                f"Metadata: {json.dumps(safe_metadata)}\n"
            )

            report: ExplainabilityReport = _run_agent_with_fallback(
                explainer,
                explainability_input,
                max_turns=3,
            )
            result.explainability = report
            llm_calls_used += 1
            result.llm_call_count += 1
            tl.log_phase_end("explainability", "Report generated")

        except Exception as exc:
            logger.warning("Explainability agent failed: %s", exc)
            tl.log_phase_end("explainability", f"FAILED: {exc}")
            result.explainability = ExplainabilityReport(
                detected_issues=f"Issues detected: {', '.join(issues) or 'none'}.",
                corrective_measures="Standard deterministic recommendations applied.",
                enhancement_applied=f"Applied: {', '.join(applied_ops) or 'none'}.",
                validation_outcome="See validation metrics table in report.",
                limitations=(
                    "This tool is for quality assurance research only and is "
                    "NOT intended for clinical diagnosis or patient care decisions."
                ),
                image_summary="Unable to generate detailed summary due to LLM error.",
                actionable_suggestions=["Re-run with --verbose for diagnostics."],
                next_steps=["Review the metrics table manually."],
            )

    result.agent_traces = tl.get_traces()
    clear_image_store()
    return result
