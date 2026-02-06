"""Tests for deterministic detection and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.metrics import compute_metrics, THRESHOLDS
from pipeline.dicom_io import normalize_image


def detect_issues(metrics: dict) -> list[str]:
    """Replicate the detection logic for tests."""
    issues = []
    if metrics["sigma"] > THRESHOLDS["noise_sigma"]:
        issues.append("noise")
    if metrics["lap_var"] < THRESHOLDS["blur_lap_var"]:
        issues.append("blur")
    if metrics["std"] < THRESHOLDS["low_contrast_std"]:
        issues.append("low_contrast")
    if metrics["pct_low"] > THRESHOLDS["clip_pct"]:
        issues.append("clipping_low")
    if metrics["pct_high"] > THRESHOLDS["clip_pct"]:
        issues.append("clipping_high")
    return issues


class TestComputeMetrics:
    def test_returns_all_keys(self, synthetic_image_clean: np.ndarray):
        metrics = compute_metrics(synthetic_image_clean)
        expected_keys = {"sigma", "lap_var", "std", "pct_low", "pct_high",
                         "entropy", "edge_density", "gradient_mag_mean",
                         "gradient_mag_std", "snr_proxy", "cnr_proxy",
                         "laplacian_energy", "histogram_spread"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_values_are_finite(self, synthetic_image_clean: np.ndarray):
        metrics = compute_metrics(synthetic_image_clean)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_noisy_image_high_sigma(self, synthetic_image_noisy: np.ndarray):
        metrics = compute_metrics(synthetic_image_noisy)
        assert metrics["sigma"] > THRESHOLDS["noise_sigma"]


class TestDetectIssues:
    def test_no_issues_for_normal_metrics(self, sample_metrics_normal: dict):
        issues = detect_issues(sample_metrics_normal)
        assert issues == []

    def test_noise_detected(self):
        metrics = {
            "sigma": 0.15,
            "lap_var": 0.01,
            "std": 0.25,
            "pct_low": 0.005,
            "pct_high": 0.005,
        }
        issues = detect_issues(metrics)
        assert "noise" in issues
        assert "blur" not in issues

    def test_blur_detected(self):
        metrics = {
            "sigma": 0.03,
            "lap_var": 0.0005,
            "std": 0.25,
            "pct_low": 0.005,
            "pct_high": 0.005,
        }
        issues = detect_issues(metrics)
        assert "blur" in issues

    def test_low_contrast_detected(self):
        metrics = {
            "sigma": 0.03,
            "lap_var": 0.01,
            "std": 0.05,
            "pct_low": 0.005,
            "pct_high": 0.005,
        }
        issues = detect_issues(metrics)
        assert "low_contrast" in issues

    def test_clipping_detected(self):
        metrics = {
            "sigma": 0.03,
            "lap_var": 0.01,
            "std": 0.25,
            "pct_low": 0.05,
            "pct_high": 0.05,
        }
        issues = detect_issues(metrics)
        assert "clipping_low" in issues
        assert "clipping_high" in issues

    def test_all_issues_detected(self, sample_metrics_all_bad: dict):
        issues = detect_issues(sample_metrics_all_bad)
        assert "noise" in issues
        assert "blur" in issues
        assert "low_contrast" in issues
        assert "clipping_low" in issues
        assert "clipping_high" in issues


class TestNormalizeImage:
    def test_output_range(self):
        img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        normed = normalize_image(img)
        assert float(np.min(normed)) == pytest.approx(0.0)
        assert float(np.max(normed)) == pytest.approx(1.0)

    def test_constant_image(self):
        img = np.full((8, 8), 42.0, dtype=np.float32)
        normed = normalize_image(img)
        assert float(np.max(normed)) == 0.0
