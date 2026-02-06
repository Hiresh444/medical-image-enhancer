from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from utils import (
    compute_metrics,
    detect_issues,
    apply_enhancements,
    compute_validation,
    build_markdown_report,
)


@dataclass
class DetectionResult:
    metrics: Dict[str, float]
    issues: List[str]


@dataclass
class RecommendationResult:
    recommendations: List[str]
    mapping: Dict[str, str]


@dataclass
class EnhancementResult:
    image: np.ndarray
    applied_ops: List[str]
    metrics: Dict[str, float]


@dataclass
class ValidationResult:
    ssim: float
    psnr: float
    quality_improvement: float
    meets_ssim: bool
    meets_psnr: bool
    meets_improvement: bool
    passes: bool
    status: str
    notes: List[str]
    # No-reference metrics
    niqe_before: float = 0.0
    niqe_after: float = 0.0
    niqe_improved: bool = True
    # Component gains
    contrast_gain: float = 0.0
    sharpness_gain: float = 0.0
    noise_change: float = 0.0


class QualityDetectionAgent:
    def run(self, image: np.ndarray) -> DetectionResult:
        metrics = compute_metrics(image)
        issues = detect_issues(metrics)
        return DetectionResult(metrics=metrics, issues=issues)


class RecommendationAgent:
    ISSUE_TO_ACTION = {
        "noise": "Apply wavelet denoising to reduce noise.",
        "low_contrast": "Apply CLAHE to improve contrast.",
        "blur": "Apply unsharp masking to improve sharpness.",
        "clipping_low": "Apply CLAHE and mild gamma correction to lift shadows.",
        "clipping_high": "Apply CLAHE and mild gamma correction to reduce highlights.",
    }

    def run(self, detection: DetectionResult) -> RecommendationResult:
        if not detection.issues:
            return RecommendationResult(
                recommendations=["No issues detected. Enhancement not required."],
                mapping={},
            )

        mapping = {
            issue: self.ISSUE_TO_ACTION.get(issue, "Review manually.")
            for issue in detection.issues
        }
        recommendations = list(mapping.values())
        return RecommendationResult(recommendations=recommendations, mapping=mapping)


class EnhancementAgent:
    def run(self, image: np.ndarray, recommendations: RecommendationResult) -> EnhancementResult:
        enhanced, applied_ops = apply_enhancements(image, list(recommendations.mapping.keys()))
        metrics = compute_metrics(enhanced)
        return EnhancementResult(image=enhanced, applied_ops=applied_ops, metrics=metrics)


class ValidationAgent:
    def run(
        self, original: np.ndarray, enhanced: np.ndarray, detection: DetectionResult
    ) -> ValidationResult:
        validation = compute_validation(original, enhanced)

        notes = []
        passes = validation["passes"]
        meets_improvement = validation["meets_improvement"]

        if not detection.issues:
            notes.append("No issues detected; enhancement not required.")
            passes = validation["meets_ssim"] and validation["meets_psnr"]
            meets_improvement = True

        status = "PASS" if passes else "FAIL"
        if status == "FAIL" and validation["quality_improvement"] > 0:
            status = "WARN"
            notes.append("Some improvement observed, but thresholds not fully met.")
        
        # Add notes about no-reference metrics
        if validation.get("niqe_improved"):
            notes.append("Naturalness preserved (NIQE-approx stable or improved).")
        else:
            notes.append("Warning: Naturalness may be degraded (possible over-processing).")
        
        if validation.get("noise_change", 0) > 0.5:
            notes.append(f"Note: Noise increased by {validation['noise_change']*100:.1f}% (sharpening side-effect).")

        return ValidationResult(
            ssim=validation["ssim"],
            psnr=validation["psnr"],
            quality_improvement=validation["quality_improvement"],
            meets_ssim=validation["meets_ssim"],
            meets_psnr=validation["meets_psnr"],
            meets_improvement=meets_improvement,
            passes=passes,
            status=status,
            notes=notes,
            niqe_before=validation.get("niqe_before", 0.0),
            niqe_after=validation.get("niqe_after", 0.0),
            niqe_improved=validation.get("niqe_improved", True),
            contrast_gain=validation.get("contrast_gain", 0.0),
            sharpness_gain=validation.get("sharpness_gain", 0.0),
            noise_change=validation.get("noise_change", 0.0),
        )


class ReportAgent:
    def run(self, context: Dict[str, object]) -> str:
        return build_markdown_report(context)
