"""Tests for pipeline.metrics – expanded metric suite."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.metrics import (
    compute_metrics,
    compute_validation,
    compute_objective_score,
    compute_edge_ratio,
    THRESHOLDS,
)


class TestExpandedMetrics:
    """Verify the 16-metric compute_metrics output."""

    def test_all_13_keys(self, synthetic_image_clean: np.ndarray):
        m = compute_metrics(synthetic_image_clean)
        expected = {
            "sigma", "lap_var", "std", "pct_low", "pct_high",
            "entropy", "edge_density", "gradient_mag_mean",
            "gradient_mag_std", "snr_proxy", "cnr_proxy",
            "laplacian_energy", "histogram_spread",
            "local_contrast_std", "gradient_strength", "gradient_entropy",
        }
        assert expected.issubset(set(m.keys()))
        assert len(m) == 16

    def test_entropy_positive(self, synthetic_image_clean: np.ndarray):
        m = compute_metrics(synthetic_image_clean)
        assert m["entropy"] >= 0

    def test_snr_positive(self, synthetic_image_clean: np.ndarray):
        m = compute_metrics(synthetic_image_clean)
        assert m["snr_proxy"] >= 0

    def test_noisy_lower_snr(
        self,
        synthetic_image_clean: np.ndarray,
        synthetic_image_noisy: np.ndarray,
    ):
        clean = compute_metrics(synthetic_image_clean)
        noisy = compute_metrics(synthetic_image_noisy)
        assert clean["snr_proxy"] > noisy["snr_proxy"]


class TestValidation:
    def test_ssim_fields(self, synthetic_image_clean: np.ndarray):
        v = compute_validation(synthetic_image_clean, synthetic_image_clean)
        for key in ("ssim", "psnr", "niqe_before", "niqe_after", "passes",
                     "entropy_before", "entropy_after",
                     "snr_before", "snr_after", "cnr_before", "cnr_after"):
            assert key in v, f"Missing key: {key}"

    def test_identical_passes(self, synthetic_image_clean: np.ndarray):
        v = compute_validation(synthetic_image_clean, synthetic_image_clean)
        assert v["passes"] is True
        assert v["ssim"] == pytest.approx(1.0)


class TestObjectiveScore:
    def test_range(self, synthetic_image_clean: np.ndarray):
        v = compute_validation(synthetic_image_clean, synthetic_image_clean)
        score, breakdown = compute_objective_score(v)
        assert isinstance(score, float)
        assert isinstance(breakdown, dict)


class TestEdgeRatio:
    def test_returns_float(self, synthetic_image_clean: np.ndarray):
        ratio = compute_edge_ratio(synthetic_image_clean)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0
