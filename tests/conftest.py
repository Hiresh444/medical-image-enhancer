"""Shared test fixtures for the mdimg test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_image_clean() -> np.ndarray:
    """64×64 smooth gradient — should trigger no issues."""
    rng = np.random.default_rng(42)
    gradient = np.linspace(0.1, 0.9, 64 * 64).reshape(64, 64).astype(np.float32)
    # Add tiny noise so sigma estimator doesn't return 0
    gradient += rng.normal(0, 0.005, gradient.shape).astype(np.float32)
    return np.clip(gradient, 0.0, 1.0)


@pytest.fixture
def synthetic_image_noisy() -> np.ndarray:
    """64×64 image with heavy Gaussian noise — should trigger 'noise'."""
    rng = np.random.default_rng(99)
    base = np.full((64, 64), 0.5, dtype=np.float32)
    noise = rng.normal(0, 0.15, base.shape).astype(np.float32)
    return np.clip(base + noise, 0.0, 1.0)


@pytest.fixture
def synthetic_image_low_contrast() -> np.ndarray:
    """64×64 image with very low dynamic range — should trigger 'low_contrast'."""
    return np.full((64, 64), 0.5, dtype=np.float32) + np.float32(0.01) * np.random.default_rng(7).standard_normal((64, 64)).astype(np.float32)


@pytest.fixture
def sample_metrics_normal() -> dict[str, float]:
    """Metrics dict that should produce NO issues."""
    return {
        "sigma": 0.03,
        "lap_var": 0.01,
        "std": 0.25,
        "pct_low": 0.005,
        "pct_high": 0.005,
    }


@pytest.fixture
def sample_metrics_all_bad() -> dict[str, float]:
    """Metrics dict that should trigger ALL issue types."""
    return {
        "sigma": 0.15,
        "lap_var": 0.0005,
        "std": 0.05,
        "pct_low": 0.05,
        "pct_high": 0.05,
    }


@pytest.fixture
def sample_enhancement_plan_dict() -> dict:
    """Valid EnhancementPlan as a raw dict."""
    return {
        "recommended_ops": ["denoise", "clahe", "gamma", "unsharp"],
        "params": {
            "clahe_clip_limit": 0.02,
            "clahe_tile_size": 16,
            "gamma": 0.95,
            "unsharp_radius": 0.8,
            "unsharp_amount": 0.5,
            "denoise_mode": "soft",
            "post_denoise_strength": 0.2,
        },
        "risk_warnings": ["Mild halo risk from unsharp mask"],
        "rationale": "Image shows noise and low contrast; conservative plan.",
        "safety": "Do not apply aggressive sharpening.",
        "stop_reason": None,
    }
