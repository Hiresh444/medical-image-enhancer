"""Tests for the full pipeline: deterministic path, parameterised enhancement,
and parameter clamping.  No real DICOM files or OpenAI API calls required."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.schemas import EnhancementPlan, EnhancementParams
from pipeline.enhancement import (
    apply_enhancements,
    apply_enhancements_from_params,
)
from pipeline.metrics import compute_validation


class TestApplyEnhancements:
    """Existing deterministic enhancement path."""

    def test_returns_valid_image(self, synthetic_image_noisy: np.ndarray):
        enhanced, ops = apply_enhancements(synthetic_image_noisy, ["noise"])
        assert enhanced.shape == synthetic_image_noisy.shape
        assert enhanced.dtype == np.float32
        assert float(np.min(enhanced)) >= 0.0
        assert float(np.max(enhanced)) <= 1.0
        assert len(ops) > 0

    def test_no_ops_when_no_issues(self, synthetic_image_clean: np.ndarray):
        enhanced, ops = apply_enhancements(synthetic_image_clean, [])
        assert len(ops) == 0
        np.testing.assert_array_equal(enhanced, synthetic_image_clean)


class TestApplyEnhancementsFromParams:
    """New parameterised enhancement path."""

    def test_basic_plan(self, synthetic_image_noisy: np.ndarray, sample_enhancement_plan_dict: dict):
        plan = EnhancementPlan(**sample_enhancement_plan_dict)
        enhanced, ops = apply_enhancements_from_params(synthetic_image_noisy, plan)
        assert enhanced.shape == synthetic_image_noisy.shape
        assert enhanced.dtype == np.float32
        assert 0.0 <= float(np.min(enhanced))
        assert float(np.max(enhanced)) <= 1.0
        assert len(ops) > 0

    def test_empty_ops(self, synthetic_image_clean: np.ndarray):
        plan = EnhancementPlan(
            recommended_ops=[],
            stop_reason="No issues.",
        )
        enhanced, ops = apply_enhancements_from_params(synthetic_image_clean, plan)
        assert len(ops) == 0

    def test_parameter_clamping(self, synthetic_image_noisy: np.ndarray):
        """Out-of-range params should be silently clamped."""
        plan = EnhancementPlan(
            recommended_ops=["clahe", "unsharp"],
            params=EnhancementParams(
                clahe_clip_limit=999.0,   # way above max (0.05)
                unsharp_amount=-5.0,      # below min (0.1)
            ),
        )
        enhanced, ops = apply_enhancements_from_params(synthetic_image_noisy, plan)
        # Should NOT crash; values are clamped internally
        assert enhanced.shape == synthetic_image_noisy.shape
        assert len(ops) >= 1

    def test_invalid_denoise_mode_defaults_to_soft(
        self, synthetic_image_noisy: np.ndarray,
    ):
        plan = EnhancementPlan(
            recommended_ops=["denoise"],
            params=EnhancementParams(denoise_mode="INVALID"),
        )
        enhanced, ops = apply_enhancements_from_params(synthetic_image_noisy, plan)
        assert any("soft" in op for op in ops)


class TestComputeValidation:
    def test_identical_images(self, synthetic_image_clean: np.ndarray):
        result = compute_validation(synthetic_image_clean, synthetic_image_clean)
        assert result["ssim"] == pytest.approx(1.0)
        assert result["passes"] is True

    def test_enhanced_vs_original(self, synthetic_image_noisy: np.ndarray):
        enhanced, _ = apply_enhancements(synthetic_image_noisy, ["noise"])
        result = compute_validation(synthetic_image_noisy, enhanced)
        assert "ssim" in result
        assert "psnr" in result
        assert "niqe_before" in result
        assert isinstance(result["passes"], bool)


class TestDeterministicPathEndToEnd:
    """Smoke test: run the full deterministic pipeline on a synthetic image."""

    def test_full_pipeline_no_crash(self, synthetic_image_noisy: np.ndarray):
        from pipeline.core_agents import (
            QualityDetectionAgent,
            RecommendationAgent,
            EnhancementAgent,
            ValidationAgent,
            ReportAgent,
        )

        detector = QualityDetectionAgent()
        detection = detector.run(synthetic_image_noisy)
        assert isinstance(detection.issues, list)
        assert isinstance(detection.metrics, dict)

        recommender = RecommendationAgent()
        recommendations = recommender.run(detection)

        enhancer = EnhancementAgent()
        enhancement = enhancer.run(synthetic_image_noisy, recommendations)
        assert enhancement.image.shape == synthetic_image_noisy.shape

        validator = ValidationAgent()
        validation = validator.run(synthetic_image_noisy, enhancement.image, detection)
        assert validation.status in ("PASS", "WARN", "FAIL")

        reporter = ReportAgent()
        context = {
            "input_path": "test.dcm",
            "metadata": {"Modality": "CT"},
            "issues": detection.issues,
            "recommendations": recommendations.recommendations,
            "applied_ops": enhancement.applied_ops,
            "metrics_before": detection.metrics,
            "metrics_after": enhancement.metrics,
            "validation": validation,
            "visuals": {},
            "notes": validation.notes,
        }
        report = reporter.run(context)
        assert "# 🧪" in report
        assert "SSIM" in report
