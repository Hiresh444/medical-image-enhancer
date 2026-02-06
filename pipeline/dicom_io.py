"""DICOM I/O, normalisation, visualisation, and markdown report builder.

Handles loading DICOM files, converting to grayscale, normalising to [0, 1],
saving before/after visualisations, and generating the markdown QA report.
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixel_data_handlers.util import apply_modality_lut

from pipeline.metrics import THRESHOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------


def load_dicom(path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    """Load a DICOM file, returning (pixel_array, safe_metadata)."""
    try:
        ds = pydicom.dcmread(path)
    except (InvalidDicomError, FileNotFoundError) as exc:
        raise ValueError("Invalid or missing DICOM file.") from exc

    if "PixelData" not in ds:
        raise ValueError("DICOM file does not contain pixel data.")

    try:
        image = ds.pixel_array
    except Exception as exc:
        raise ValueError("Unable to decode DICOM pixel data.") from exc

    image = apply_modality_lut(image, ds).astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image = _to_grayscale(image)

    metadata = {
        "Modality": str(getattr(ds, "Modality", "Unknown")),
        "BodyPartExamined": str(getattr(ds, "BodyPartExamined", "Unknown")),
        "StudyDescription": str(getattr(ds, "StudyDescription", "Unknown")),
    }

    return image, metadata


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert multi-channel or multi-frame image to 2-D grayscale."""
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            rgb = image[..., :3]
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if image.shape[0] in (3, 4):
            rgb = image[:3, ...]
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        mid = image.shape[0] // 2
        return image[mid]

    if image.ndim > 3:
        while image.ndim > 2:
            mid = image.shape[0] // 2
            image = image[mid]
        return image

    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1]."""
    image = image.astype(np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def save_visuals(
    original: np.ndarray,
    enhanced: np.ndarray,
    out_dir: str,
    base_name: str,
) -> Dict[str, str]:
    """Save a side-by-side before/after comparison PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    figure_path = os.path.join(out_dir, f"{base_name}_before_after.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title("After")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return {"before_after": figure_path}


def save_single_image(
    image: np.ndarray, out_path: str, title: str = ""
) -> str:
    """Save a single grayscale image to *out_path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def build_markdown_report(context: Dict[str, object]) -> str:
    """Build a full markdown QA report from pipeline context."""
    issues = context.get("issues", [])
    recommendations = context.get("recommendations", [])
    applied_ops = context.get("applied_ops", [])
    metrics_before = context.get("metrics_before", {})
    metrics_after = context.get("metrics_after", {})
    validation = context.get("validation")
    visuals = context.get("visuals", {})
    notes = context.get("notes", [])

    status_emoji = "✅" if validation.status == "PASS" else "⚠️"
    if validation.status == "FAIL":
        status_emoji = "❌"

    ssim_str = f"{validation.ssim:.3f}"
    if np.isinf(validation.psnr):
        psnr_str = "inf"
    else:
        psnr_str = f"{validation.psnr:.2f} dB"

    lines: list[str] = []
    lines.append("# 🧪 Multi-Agent Medical Imaging QA Report")
    lines.append("")
    lines.append(f"**Input:** `{context.get('input_path', '')}`")
    lines.append(f"**Status:** {status_emoji} {validation.status}")
    lines.append("")

    metadata = context.get("metadata", {})
    if metadata:
        lines.append("## 🗂️ DICOM Metadata (Non-PHI)")
        for key, value in metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    lines.append("## 🔍 Detected Issues")
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("No issues detected.")
    lines.append("")

    lines.append("## 💡 Recommendations")
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    lines.append("## 🛠️ Applied Enhancements")
    if applied_ops:
        for op in applied_ops:
            lines.append(f"- {op}")
    else:
        lines.append("No enhancements applied.")
    lines.append("")

    # --- Quality Metrics Table (expanded) ---
    lines.append("## 📊 Quality Metrics")
    lines.append("| Metric | Before | After |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Noise σ | {metrics_before.get('sigma', 0.0):.4f} | {metrics_after.get('sigma', 0.0):.4f} |"
    )
    lines.append(
        f"| Laplacian Var | {metrics_before.get('lap_var', 0.0):.6f} | {metrics_after.get('lap_var', 0.0):.6f} |"
    )
    lines.append(
        f"| Contrast (std) | {metrics_before.get('std', 0.0):.4f} | {metrics_after.get('std', 0.0):.4f} |"
    )
    lines.append(
        f"| Clip Low (%) | {metrics_before.get('pct_low', 0.0) * 100:.2f} | {metrics_after.get('pct_low', 0.0) * 100:.2f} |"
    )
    lines.append(
        f"| Clip High (%) | {metrics_before.get('pct_high', 0.0) * 100:.2f} | {metrics_after.get('pct_high', 0.0) * 100:.2f} |"
    )
    lines.append(
        f"| Entropy | {metrics_before.get('entropy', 0.0):.3f} | {metrics_after.get('entropy', 0.0):.3f} |"
    )
    lines.append(
        f"| Edge Density | {metrics_before.get('edge_density', 0.0):.4f} | {metrics_after.get('edge_density', 0.0):.4f} |"
    )
    lines.append(
        f"| Grad. Mag Mean | {metrics_before.get('gradient_mag_mean', 0.0):.4f} | {metrics_after.get('gradient_mag_mean', 0.0):.4f} |"
    )
    lines.append(
        f"| SNR Proxy | {metrics_before.get('snr_proxy', 0.0):.2f} | {metrics_after.get('snr_proxy', 0.0):.2f} |"
    )
    lines.append(
        f"| CNR Proxy | {metrics_before.get('cnr_proxy', 0.0):.2f} | {metrics_after.get('cnr_proxy', 0.0):.2f} |"
    )
    lines.append(
        f"| Laplacian Energy | {metrics_before.get('laplacian_energy', 0.0):.6f} | {metrics_after.get('laplacian_energy', 0.0):.6f} |"
    )
    lines.append(
        f"| Histogram Spread | {metrics_before.get('histogram_spread', 0.0):.4f} | {metrics_after.get('histogram_spread', 0.0):.4f} |"
    )
    lines.append("")

    # --- Validation ---
    lines.append("## ✅ Validation")
    lines.append(f"- SSIM: {ssim_str} (>= {THRESHOLDS['ssim']})")
    lines.append(f"- PSNR: {psnr_str} (>= {THRESHOLDS['psnr']} dB)")
    lines.append(
        f"- Quality Improvement: {validation.quality_improvement:.2f} (>= {THRESHOLDS['quality_improvement']})"
    )
    lines.append("")

    # Component gains breakdown
    lines.append("### 📈 Enhancement Gains")
    lines.append("| Component | Change |")
    lines.append("| --- | --- |")
    contrast_pct = getattr(validation, "contrast_gain", 0) * 100
    sharpness_pct = getattr(validation, "sharpness_gain", 0) * 100
    noise_pct = getattr(validation, "noise_change", 0) * 100
    lines.append(
        f"| Contrast | {'+' if contrast_pct >= 0 else ''}{contrast_pct:.1f}% |"
    )
    lines.append(
        f"| Sharpness | {'+' if sharpness_pct >= 0 else ''}{sharpness_pct:.1f}% |"
    )
    lines.append(
        f"| Noise | {'+' if noise_pct >= 0 else ''}{noise_pct:.1f}% |"
    )
    lines.append("")

    # NIQE
    niqe_before = getattr(validation, "niqe_before", 0)
    niqe_after = getattr(validation, "niqe_after", 0)
    niqe_emoji = "✅" if getattr(validation, "niqe_improved", True) else "⚠️"
    lines.append("### 🎯 No-Reference Quality (NIQE-approx)")
    lines.append(f"- Before: {niqe_before:.3f}")
    lines.append(f"- After: {niqe_after:.3f}")
    lines.append(
        f"- Naturalness: {niqe_emoji} {'Preserved' if niqe_after <= niqe_before else 'Degraded'}"
    )
    lines.append("")

    # Interpretation note
    lines.append("### ℹ️ Metrics Interpretation")
    lines.append(
        "> **Note:** Full-reference metrics (SSIM, PSNR) compare enhanced image to original. "
        "For enhancement tasks, these metrics are *expected* to be lower than typical "
        "compression/reconstruction thresholds because enhancement intentionally modifies "
        "pixel values to improve visibility. The thresholds above are calibrated for "
        "*conservative enhancement* that preserves anatomical fidelity while allowing "
        "clinically meaningful improvements in contrast and sharpness."
    )
    lines.append("")
    lines.append(
        "> **NIQE-approx** is a no-reference metric estimating image naturalness. "
        "Lower values indicate more natural-looking images. An increase may suggest "
        "over-processing (halos, artifacts, or unnatural textures)."
    )
    lines.append("")

    if visuals.get("before_after"):
        lines.append("## 🖼️ Before vs After")
        lines.append(f"![Before vs After]({visuals['before_after']})")
        lines.append("")

    if notes:
        lines.append("## 📝 Notes")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # ------------------------------------------------------------------ #
    # GenAI-specific sections                                              #
    # ------------------------------------------------------------------ #
    genai_plan = context.get("genai_plan")
    if genai_plan is not None:
        lines.append("## 🤖 GenAI Plan (JSON)")
        lines.append("")
        lines.append("```json")
        if hasattr(genai_plan, "model_dump_json"):
            lines.append(genai_plan.model_dump_json(indent=2))
        else:
            lines.append(_json.dumps(genai_plan, indent=2, default=str))
        lines.append("```")
        lines.append("")

    genai_iterations = context.get("genai_iterations", [])
    if genai_iterations:
        lines.append("## 🔄 Agentic Iterations")
        lines.append("")
        lines.append(
            "| Iteration | Score | SSIM | PSNR | Quality Improvement | Chosen |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for rec in genai_iterations:
            m = rec.metrics if hasattr(rec, "metrics") else rec.get("metrics", {})
            chosen = rec.chosen if hasattr(rec, "chosen") else rec.get("chosen", False)
            score = rec.score if hasattr(rec, "score") else rec.get("score", 0)
            it = (
                rec.iteration
                if hasattr(rec, "iteration")
                else rec.get("iteration", "?")
            )
            lines.append(
                f"| {it} | {score:.4f} "
                f"| {m.get('ssim', 0):.3f} "
                f"| {m.get('psnr', 0):.2f} dB "
                f"| {m.get('quality_improvement', 0):.3f} "
                f"| {'✅' if chosen else '—'} |"
            )
        lines.append("")

    genai_model = context.get("genai_model")
    if genai_model:
        lines.append("## ⚙️ Model & Settings")
        lines.append(f"- **Model:** {genai_model}")
        lines.append(
            f"- **Max iterations:** {context.get('genai_max_iters', 'N/A')}"
        )
        lines.append(
            f"- **LLM calls:** {context.get('genai_llm_calls', 'N/A')}"
        )
        lines.append("")

    genai_prompts = context.get("genai_prompts", [])
    if genai_prompts:
        lines.append("## 📜 Prompts Used")
        for i, prompt_label in enumerate(genai_prompts, 1):
            lines.append(f"{i}. {prompt_label}")
        lines.append("")

    genai_explainability = context.get("genai_explainability")
    if genai_explainability is not None:
        lines.append("## 🧠 Explainability (GenAI)")
        lines.append("")
        if hasattr(genai_explainability, "detected_issues"):
            lines.append(
                f"**Detected Issues:** {genai_explainability.detected_issues}"
            )
            lines.append("")
            lines.append(
                f"**Corrective Measures:** {genai_explainability.corrective_measures}"
            )
            lines.append("")
            lines.append(
                f"**Enhancement Applied:** {genai_explainability.enhancement_applied}"
            )
            lines.append("")
            lines.append(
                f"**Validation Outcome:** {genai_explainability.validation_outcome}"
            )
            lines.append("")
            lines.append(
                f"**Limitations:** {genai_explainability.limitations}"
            )
            lines.append("")

            # Richer outputs
            if getattr(genai_explainability, "image_summary", ""):
                lines.append(
                    f"**Image Summary:** {genai_explainability.image_summary}"
                )
                lines.append("")

            suggestions = getattr(
                genai_explainability, "actionable_suggestions", []
            )
            if suggestions:
                lines.append("**Actionable Suggestions:**")
                for s in suggestions:
                    lines.append(f"- {s}")
                lines.append("")

            next_steps = getattr(genai_explainability, "next_steps", [])
            if next_steps:
                lines.append("**Next Steps:**")
                for s in next_steps:
                    lines.append(f"- {s}")
                lines.append("")
        else:
            lines.append(str(genai_explainability))
        lines.append("")

    # Safety / privacy statement
    if genai_plan is not None or genai_model:
        lines.append("## 🔒 Safety / Privacy")
        lines.append("")
        lines.append(
            "> **No raw images or PHI were sent to the LLM.** Only numeric "
            "quality metrics (σ, Laplacian variance, contrast std, clipping "
            "percentages) and non-PHI DICOM metadata (Modality, "
            "BodyPartExamined, StudyDescription) were transmitted to the "
            "language model. All image processing was executed locally."
        )
        lines.append("")

    return "\n".join(lines)
