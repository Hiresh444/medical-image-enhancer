"""Quality metrics computation for medical image enhancement.

Provides both full-reference (SSIM, PSNR) and no-reference metrics (entropy,
edge density, SNR/CNR proxies, Laplacian energy, histogram spread, NIQE-approx).
All functions operate on normalised [0, 1] float32 images.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.ndimage import uniform_filter
from skimage import filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "noise_sigma": 0.08,
    "blur_lap_var": 0.001,
    "low_contrast_std": 0.12,
    "clip_pct": 0.01,
    # Relaxed full-reference thresholds for enhancement tasks
    "ssim": 0.70,
    "psnr": 22.0,
    "quality_improvement": 0.10,
}


# ---------------------------------------------------------------------------
# Core quality metrics
# ---------------------------------------------------------------------------


def compute_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive quality metrics for a normalised image.

    Returns the original 5 metrics plus 7 new ones for a total of 12.
    """
    sigma = float(estimate_sigma(image, channel_axis=None, average_sigmas=True))
    lap = filters.laplace(image)
    lap_var = float(np.var(lap))
    std = float(np.std(image))
    pct_low = float(np.mean(image <= 0.01))
    pct_high = float(np.mean(image >= 0.99))

    # --- New metrics ---
    # Entropy (Shannon): higher = more information content
    entropy = _shannon_entropy(image)

    # Edge density: fraction of edge pixels (Sobel-based)
    edge_density = _edge_density(image)

    # Gradient magnitude statistics (Sobel)
    grad_mag = np.sqrt(filters.sobel_h(image) ** 2 + filters.sobel_v(image) ** 2)
    gradient_mag_mean = float(np.mean(grad_mag))
    gradient_mag_std = float(np.std(grad_mag))

    # SNR proxy: mean / noise_sigma
    snr_proxy = float(np.mean(image) / max(sigma, 1e-8))

    # CNR proxy: (p95 - p05) / noise_sigma
    p05, p95 = float(np.percentile(image, 5)), float(np.percentile(image, 95))
    cnr_proxy = float((p95 - p05) / max(sigma, 1e-8))

    # Laplacian energy: mean of squared Laplacian (sharpness energy)
    laplacian_energy = float(np.mean(lap ** 2))

    # Histogram spread: interquartile range of pixel intensities
    q25, q75 = float(np.percentile(image, 25)), float(np.percentile(image, 75))
    histogram_spread = q75 - q25

    # --- Extra lightweight metrics ---
    # Local contrast std: std of local standard deviations (7x7 patches)
    local_contrast_std = _local_contrast_std(image)

    # Gradient strength: mean of top-10% gradient magnitudes (strong edges)
    gradient_strength = _gradient_strength(grad_mag)

    # Gradient entropy: Shannon entropy of gradient magnitude histogram
    gradient_entropy = _gradient_entropy(grad_mag)

    return {
        "sigma": sigma,
        "lap_var": lap_var,
        "std": std,
        "pct_low": pct_low,
        "pct_high": pct_high,
        # New
        "entropy": entropy,
        "edge_density": edge_density,
        "gradient_mag_mean": gradient_mag_mean,
        "gradient_mag_std": gradient_mag_std,
        "snr_proxy": snr_proxy,
        "cnr_proxy": cnr_proxy,
        "laplacian_energy": laplacian_energy,
        "histogram_spread": histogram_spread,
        # Extra lightweight
        "local_contrast_std": local_contrast_std,
        "gradient_strength": gradient_strength,
        "gradient_entropy": gradient_entropy,
    }


def _shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Compute Shannon entropy of the image histogram."""
    hist, _ = np.histogram(image.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def _local_contrast_std(image: np.ndarray, patch_size: int = 7) -> float:
    """Standard deviation of local standard deviations (patch-based).

    Captures local texture quality beyond global ``std``.
    """
    local_mean = uniform_filter(image, size=patch_size)
    local_sq_mean = uniform_filter(image ** 2, size=patch_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    return float(np.std(local_std))


def _gradient_strength(grad_mag: np.ndarray) -> float:
    """Mean of the top-10% gradient magnitudes (strong edge strength)."""
    threshold = float(np.percentile(grad_mag, 90))
    strong = grad_mag[grad_mag >= threshold]
    if strong.size == 0:
        return 0.0
    return float(np.mean(strong))


def _gradient_entropy(grad_mag: np.ndarray, bins: int = 128) -> float:
    """Shannon entropy of the gradient magnitude histogram.

    Over-sharpened images have concentrated gradient distributions (low entropy).
    """
    hist, _ = np.histogram(grad_mag.ravel(), bins=bins, range=(0.0, float(grad_mag.max()) + 1e-8))
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def _edge_density(image: np.ndarray, threshold_frac: float = 0.1) -> float:
    """Fraction of pixels with gradient magnitude above *threshold_frac* of max."""
    grad_mag = np.sqrt(filters.sobel_h(image) ** 2 + filters.sobel_v(image) ** 2)
    threshold = threshold_frac * grad_mag.max() if grad_mag.max() > 0 else 0
    return float(np.mean(grad_mag > threshold))


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------


def detect_issues(metrics: Dict[str, float]) -> list[str]:
    """Detect quality issues from metrics by comparing against thresholds."""
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


# ---------------------------------------------------------------------------
# NIQE-approximation (no-reference naturalness score; lower = better)
# ---------------------------------------------------------------------------


def compute_niqe_approximation(image: np.ndarray) -> float:
    """Approximate no-reference quality score (lower = better).

    Measures local-variance consistency, edge coherence, and halo presence.
    """
    patch_size = 16

    # Local mean and variance
    local_mean = uniform_filter(image, size=patch_size)
    local_sq_mean = uniform_filter(image ** 2, size=patch_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # Coefficient of variation of local variance
    var_of_var = float(np.std(local_var) / (np.mean(local_var) + 1e-8))

    # Edge coherence: Laplacian-to-gradient ratio (halos push this >1.0)
    laplacian = np.abs(filters.laplace(image))
    gradient_mag = np.sqrt(filters.sobel_h(image) ** 2 + filters.sobel_v(image) ** 2)
    edge_ratio = float(np.mean(laplacian) / (np.mean(gradient_mag) + 1e-8))

    halo_penalty = max(0, edge_ratio - 1.0) * 10

    niqe_approx = var_of_var + halo_penalty
    return float(niqe_approx)


def compute_edge_ratio(image: np.ndarray) -> float:
    """Compute edge_ratio for halo detection.  Values > 1.0 suggest halos."""
    laplacian = np.abs(filters.laplace(image))
    gradient_mag = np.sqrt(filters.sobel_h(image) ** 2 + filters.sobel_v(image) ** 2)
    return float(np.mean(laplacian) / (np.mean(gradient_mag) + 1e-8))


# ---------------------------------------------------------------------------
# Full-reference validation
# ---------------------------------------------------------------------------


def compute_validation(
    original: np.ndarray, enhanced: np.ndarray
) -> Dict[str, object]:
    """Compare original vs enhanced with full- and no-reference metrics."""
    metrics_before = compute_metrics(original)
    metrics_after = compute_metrics(enhanced)

    ssim = float(structural_similarity(original, enhanced, data_range=1.0))
    psnr = float(peak_signal_noise_ratio(original, enhanced, data_range=1.0))

    niqe_before = compute_niqe_approximation(original)
    niqe_after = compute_niqe_approximation(enhanced)
    niqe_improved = niqe_after <= niqe_before

    eps = 1e-8
    contrast_gain = (metrics_after["std"] - metrics_before["std"]) / max(
        metrics_before["std"], eps
    )
    sharpness_gain = (metrics_after["lap_var"] - metrics_before["lap_var"]) / max(
        metrics_before["lap_var"], eps
    )
    noise_reduction = (metrics_before["sigma"] - metrics_after["sigma"]) / max(
        metrics_before["sigma"], eps
    )

    # New component metrics
    entropy_change = metrics_after["entropy"] - metrics_before["entropy"]
    snr_change = metrics_after["snr_proxy"] - metrics_before["snr_proxy"]
    cnr_change = metrics_after["cnr_proxy"] - metrics_before["cnr_proxy"]
    edge_density_change = metrics_after["edge_density"] - metrics_before["edge_density"]
    histogram_spread_change = (
        metrics_after["histogram_spread"] - metrics_before["histogram_spread"]
    )
    laplacian_energy_before = metrics_before["laplacian_energy"]
    laplacian_energy_after = metrics_after["laplacian_energy"]

    # Extra lightweight metric changes
    local_contrast_change = (
        metrics_after["local_contrast_std"] - metrics_before["local_contrast_std"]
    )
    gradient_strength_change = (
        metrics_after["gradient_strength"] - metrics_before["gradient_strength"]
    )
    gradient_entropy_change = (
        metrics_after["gradient_entropy"] - metrics_before["gradient_entropy"]
    )

    edge_ratio_after = compute_edge_ratio(enhanced)

    quality_improvement = float(
        0.35 * contrast_gain + 0.35 * sharpness_gain + 0.30 * noise_reduction
    )

    meets_ssim = ssim >= THRESHOLDS["ssim"]
    meets_psnr = psnr >= THRESHOLDS["psnr"]
    meets_improvement = quality_improvement >= THRESHOLDS["quality_improvement"]

    passes = (
        (meets_ssim and meets_psnr)
        or (meets_ssim and meets_improvement)
        or (meets_psnr and meets_improvement and niqe_improved)
    )

    return {
        "ssim": ssim,
        "psnr": psnr,
        "quality_improvement": quality_improvement,
        "meets_ssim": meets_ssim,
        "meets_psnr": meets_psnr,
        "meets_improvement": meets_improvement,
        "passes": passes,
        "niqe_before": niqe_before,
        "niqe_after": niqe_after,
        "niqe_improved": niqe_improved,
        "contrast_gain": contrast_gain,
        "sharpness_gain": sharpness_gain,
        "noise_change": -noise_reduction,  # Positive = noise increased
        # New fields
        "entropy_before": metrics_before["entropy"],
        "entropy_after": metrics_after["entropy"],
        "entropy_change": entropy_change,
        "snr_before": metrics_before["snr_proxy"],
        "snr_after": metrics_after["snr_proxy"],
        "snr_change": snr_change,
        "cnr_before": metrics_before["cnr_proxy"],
        "cnr_after": metrics_after["cnr_proxy"],
        "cnr_change": cnr_change,
        "edge_density_change": edge_density_change,
        "histogram_spread_change": histogram_spread_change,
        "laplacian_energy_before": laplacian_energy_before,
        "laplacian_energy_after": laplacian_energy_after,
        "edge_ratio": edge_ratio_after,
        # Extra lightweight
        "local_contrast_before": metrics_before["local_contrast_std"],
        "local_contrast_after": metrics_after["local_contrast_std"],
        "local_contrast_change": local_contrast_change,
        "gradient_strength_before": metrics_before["gradient_strength"],
        "gradient_strength_after": metrics_after["gradient_strength"],
        "gradient_strength_change": gradient_strength_change,
        "gradient_entropy_before": metrics_before["gradient_entropy"],
        "gradient_entropy_after": metrics_after["gradient_entropy"],
        "gradient_entropy_change": gradient_entropy_change,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }


# ---------------------------------------------------------------------------
# Composite objective score (used by tuning agent)
# ---------------------------------------------------------------------------


def compute_objective_score(validation: dict) -> tuple[float, dict]:
    """Compute a single scalar score from validation metrics (higher = better).

    Returns (score, breakdown_dict).
    """
    contrast_gain = float(validation.get("contrast_gain", 0))
    sharpness_gain = float(validation.get("sharpness_gain", 0))
    noise_change = float(validation.get("noise_change", 0))
    niqe_before = float(validation.get("niqe_before", 0))
    niqe_after = float(validation.get("niqe_after", 0))
    passes = bool(validation.get("passes", False))
    edge_ratio = float(validation.get("edge_ratio", 0))

    # Entropy stability: penalise large swings in either direction
    entropy_change = float(validation.get("entropy_change", 0))
    entropy_penalty = max(0.0, abs(entropy_change) - 0.5) * 2.0

    # SNR improvement reward
    snr_change = float(validation.get("snr_change", 0))
    snr_reward = max(0.0, min(snr_change * 0.1, 0.5))  # cap at 0.5

    # Histogram spread improvement (mild reward)
    hs_change = float(validation.get("histogram_spread_change", 0))
    hs_reward = max(0.0, min(hs_change * 0.5, 0.3))

    # Local contrast improvement reward
    lc_change = float(validation.get("local_contrast_change", 0))
    local_contrast_reward = max(0.0, min(lc_change * 0.3, 0.3))

    # Gradient strength improvement reward
    gs_change = float(validation.get("gradient_strength_change", 0))
    gradient_strength_reward = max(0.0, min(gs_change * 0.2, 0.2))

    # Gradient entropy stability: penalise large swings
    ge_change = float(validation.get("gradient_entropy_change", 0))
    gradient_entropy_penalty = max(0.0, abs(ge_change) - 0.3) * 1.5

    niqe_degradation = max(0.0, niqe_after - niqe_before)
    noise_penalty = max(0.0, noise_change)
    halo_penalty = max(0.0, edge_ratio - 1.0) * 5.0

    score = (
        0.35 * contrast_gain
        + 0.35 * sharpness_gain
        - 0.30 * noise_penalty
        - 5.0 * niqe_degradation
        - 10.0 * (0 if passes else 1)
        - halo_penalty
        - entropy_penalty
        + snr_reward
        + hs_reward
        + local_contrast_reward
        + gradient_strength_reward
        - gradient_entropy_penalty
    )

    breakdown = {
        "contrast_gain": round(contrast_gain, 4),
        "sharpness_gain": round(sharpness_gain, 4),
        "noise_penalty": round(noise_penalty, 4),
        "niqe_degradation": round(niqe_degradation, 4),
        "halo_penalty": round(halo_penalty, 4),
        "entropy_penalty": round(entropy_penalty, 4),
        "snr_reward": round(snr_reward, 4),
        "hs_reward": round(hs_reward, 4),
        "local_contrast_reward": round(local_contrast_reward, 4),
        "gradient_strength_reward": round(gradient_strength_reward, 4),
        "gradient_entropy_penalty": round(gradient_entropy_penalty, 4),
        "passes": passes,
    }

    return round(float(score), 4), breakdown
