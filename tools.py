"""Function-tool wrappers that expose deterministic image-processing functions
to OpenAI Agents SDK agents.

**Critical design rule**: Tools accept string image-IDs (keys into an in-memory
``_IMAGE_STORE``), *never* raw pixel arrays or DICOM data.  This guarantees no
PHI or image data is serialised into tool-call JSON / logged externally.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import numpy as np

from agents import function_tool  # openai-agents SDK

from schemas import EnhancementPlan, PARAM_BOUNDS
from utils import compute_metrics, compute_validation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory image store (module-private — never exposed to LLM)
# ---------------------------------------------------------------------------

_IMAGE_STORE: dict[str, np.ndarray] = {}


def register_image(image: np.ndarray, name: str | None = None) -> str:
    """Store an image and return its string key."""
    key = name or f"img_{uuid.uuid4().hex[:8]}"
    _IMAGE_STORE[key] = image.copy()
    return key


def get_image(image_id: str) -> np.ndarray:
    """Retrieve an image by key.  Raises ``KeyError`` if missing."""
    if image_id not in _IMAGE_STORE:
        raise KeyError(f"Image '{image_id}' not found in store.")
    return _IMAGE_STORE[image_id]


def clear_image_store() -> None:
    """Remove all images from the store (call after pipeline completes)."""
    _IMAGE_STORE.clear()


# ---------------------------------------------------------------------------
# Helper: clamp parameters to safe bounds
# ---------------------------------------------------------------------------

def _clamp_params(plan: EnhancementPlan) -> EnhancementPlan:
    """Return a copy of *plan* with all numeric parameters clamped to safe
    bounds defined in ``PARAM_BOUNDS``."""
    p = plan.params.model_copy()
    for field_name, (lo, hi) in PARAM_BOUNDS.items():
        val = getattr(p, field_name, None)
        if val is not None:
            if isinstance(val, (int, float)):
                clamped = type(val)(max(lo, min(hi, val)))
                setattr(p, field_name, clamped)
    # Ensure denoise_mode is valid
    if p.denoise_mode not in ("soft", "hard"):
        p.denoise_mode = "soft"
    return plan.model_copy(update={"params": p})


# ---------------------------------------------------------------------------
# Function tools (decorated for OpenAI Agents SDK)
# ---------------------------------------------------------------------------

@function_tool
def tool_get_metrics(image_id: str) -> str:
    """Compute quality metrics for an image stored in the image store.

    Args:
        image_id: Key of the image in the in-memory store.

    Returns:
        JSON string with keys: sigma, lap_var, std, pct_low, pct_high.
    """
    try:
        image = get_image(image_id)
        metrics = compute_metrics(image)
        return json.dumps(metrics)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@function_tool
def tool_apply_enhancement(image_id: str, plan_json: str) -> str:
    """Apply an enhancement plan to an image and store the result.

    The plan is parsed, parameters are clamped to safe bounds, and the
    deterministic enhancement pipeline is executed.  The enhanced image
    is stored under a new key returned in the response.

    Args:
        image_id:  Key of the original image in the in-memory store.
        plan_json: JSON string conforming to the EnhancementPlan schema.

    Returns:
        JSON with keys: enhanced_image_id, applied_ops, metrics.
    """
    # Lazy import to avoid circular dependency at module load time
    from utils import apply_enhancements_from_params

    try:
        image = get_image(image_id)
        plan = EnhancementPlan.model_validate_json(plan_json)
        plan = _clamp_params(plan)

        enhanced, applied_ops = apply_enhancements_from_params(image, plan)
        enhanced_id = register_image(enhanced, f"enhanced_{uuid.uuid4().hex[:6]}")
        metrics = compute_metrics(enhanced)

        return json.dumps({
            "enhanced_image_id": enhanced_id,
            "applied_ops": applied_ops,
            "metrics": metrics,
        })
    except Exception as exc:
        logger.exception("tool_apply_enhancement failed")
        return json.dumps({"error": str(exc)})


@function_tool
def tool_validate(original_id: str, enhanced_id: str) -> str:
    """Validate an enhanced image against the original.

    Computes SSIM, PSNR, NIQE-approx, component gains, and overall
    pass/fail status.

    Args:
        original_id: Key of the original image in the store.
        enhanced_id: Key of the enhanced image in the store.

    Returns:
        JSON validation dict with full-reference and no-reference metrics.
    """
    try:
        original = get_image(original_id)
        enhanced = get_image(enhanced_id)
        result = compute_validation(original, enhanced)
        # Ensure all values are JSON-serializable
        serialisable = {
            k: (float(v) if isinstance(v, (np.floating, float)) else bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in result.items()
        }
        return json.dumps(serialisable)
    except Exception as exc:
        logger.exception("tool_validate failed")
        return json.dumps({"error": str(exc)})


@function_tool
def tool_score_plan(validation_json: str) -> str:
    """Compute a scalar objective score from validation metrics.

    Higher is better.  The score balances contrast/sharpness gains against
    noise amplification and naturalness degradation.

    Args:
        validation_json: JSON string of the validation dict (from tool_validate).

    Returns:
        JSON with 'score' (float) and 'breakdown' (dict).
    """
    try:
        v: dict[str, Any] = json.loads(validation_json)
        if "error" in v:
            return json.dumps({"score": -100.0, "breakdown": {}, "error": v["error"]})

        contrast_gain = float(v.get("contrast_gain", 0))
        sharpness_gain = float(v.get("sharpness_gain", 0))
        noise_change = float(v.get("noise_change", 0))  # positive = noise up
        niqe_before = float(v.get("niqe_before", 0))
        niqe_after = float(v.get("niqe_after", 0))
        passes = bool(v.get("passes", False))

        niqe_degradation = max(0.0, niqe_after - niqe_before)
        noise_penalty = max(0.0, noise_change)  # penalise noise increase

        score = (
            0.35 * contrast_gain
            + 0.35 * sharpness_gain
            - 0.30 * noise_penalty
            - 5.0 * niqe_degradation
            - 10.0 * (0 if passes else 1)
        )

        return json.dumps({
            "score": round(float(score), 4),
            "breakdown": {
                "contrast_gain": round(contrast_gain, 4),
                "sharpness_gain": round(sharpness_gain, 4),
                "noise_penalty": round(noise_penalty, 4),
                "niqe_degradation": round(niqe_degradation, 4),
                "passes": passes,
            },
        })
    except Exception as exc:
        return json.dumps({"score": -100.0, "error": str(exc)})
