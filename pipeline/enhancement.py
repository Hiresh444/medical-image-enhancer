"""Image enhancement operations with safeguards against over-processing.

All functions operate on normalised [0, 1] float32 images.  Parameters are
safety-clamped to ``PARAM_BOUNDS`` before execution.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from scipy.ndimage import uniform_filter
from skimage import exposure, filters
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle, estimate_sigma

if TYPE_CHECKING:
    from pipeline.schemas import EnhancementPlan

from pipeline.metrics import (
    compute_metrics,
    compute_niqe_approximation,
    compute_edge_ratio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default (deterministic) enhancement parameters
# ---------------------------------------------------------------------------

ENHANCEMENT_PARAMS = {
    "clahe_clip_limit": 0.015,
    "clahe_tile_size": 16,
    "gamma_brighten": 0.95,
    "gamma_darken": 1.05,
    "unsharp_radius": 0.8,
    "unsharp_amount": 0.5,
    "denoise_sigma": None,
    "denoise_wavelet_mode": "soft",
    "post_denoise_strength": 0.3,
}


# ---------------------------------------------------------------------------
# Safeguard helpers
# ---------------------------------------------------------------------------


def _check_halo(enhanced: np.ndarray, max_edge_ratio: float = 1.5) -> bool:
    """Return True if halo artifacts are detected."""
    return compute_edge_ratio(enhanced) > max_edge_ratio


def _check_noise_amplification(
    original: np.ndarray, enhanced: np.ndarray, max_ratio: float = 1.3
) -> bool:
    """Return True if noise was amplified beyond threshold."""
    sigma_before = float(estimate_sigma(original, channel_axis=None, average_sigmas=True))
    sigma_after = float(estimate_sigma(enhanced, channel_axis=None, average_sigmas=True))
    if sigma_before < 1e-8:
        return False
    return sigma_after > sigma_before * max_ratio


def _check_over_processing(
    original: np.ndarray, enhanced: np.ndarray, max_niqe_degradation: float = 0.5
) -> bool:
    """Return True if NIQE-approx degraded beyond tolerance."""
    niqe_before = compute_niqe_approximation(original)
    niqe_after = compute_niqe_approximation(enhanced)
    return (niqe_after - niqe_before) > max_niqe_degradation


# ---------------------------------------------------------------------------
# Light denoise helper
# ---------------------------------------------------------------------------


def _light_denoise(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Apply light denoising to clean up artefacts without over-smoothing."""
    sigma_est = float(estimate_sigma(image, channel_axis=None, average_sigmas=True))
    if sigma_est < 0.001:
        return image

    denoised = denoise_wavelet(
        image,
        channel_axis=None,
        rescale_sigma=True,
        mode="soft",
        sigma=sigma_est * 0.5,
    )
    blended = (1 - strength) * image + strength * denoised
    return blended.astype(np.float32)


# ---------------------------------------------------------------------------
# Bilateral filter (edge-preserving denoise)
# ---------------------------------------------------------------------------


def _bilateral_filter(
    image: np.ndarray,
    d: int = 5,
    sigma_color: float = 0.05,
    sigma_space: float = 0.05,
) -> np.ndarray:
    """Simple bilateral-like filter using spatial Gaussian + intensity weighting.

    Implemented without OpenCV dependency — uses a sliding-window approach
    with Gaussian kernels.  Approximate but lightweight.
    """
    if d <= 0:
        return image

    # Clamp diameter to odd value
    d = min(d, 9)
    if d % 2 == 0:
        d += 1
    radius = d // 2

    padded = np.pad(image, radius, mode="reflect")
    result = np.zeros_like(image)
    weight_sum = np.zeros_like(image)

    # Spatial Gaussian weights
    y_coords, x_coords = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial_w = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma_space ** 2 * d ** 2))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                radius + dy : radius + dy + image.shape[0],
                radius + dx : radius + dx + image.shape[1],
            ]
            intensity_diff = image - shifted
            intensity_w = np.exp(-(intensity_diff ** 2) / (2 * sigma_color ** 2))
            w = spatial_w[dy + radius, dx + radius] * intensity_w
            result += w * shifted
            weight_sum += w

    result = result / (weight_sum + 1e-10)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic enhancement (issue-based: original behaviour)
# ---------------------------------------------------------------------------


def apply_enhancements(
    image: np.ndarray, issues: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Apply enhancements based on detected issues with conservative parameters.

    Pipeline order:
    1. Pre-denoise (if noisy)
    2. CLAHE (mild)
    3. Gamma correction (subtle)
    4. Unsharp mask (gentle)
    5. Post-denoise (light)
    """
    enhanced = image.copy()
    applied_ops: List[str] = []
    params = ENHANCEMENT_PARAMS

    # Step 1: Pre-denoise
    if "noise" in issues:
        enhanced = denoise_wavelet(
            enhanced,
            channel_axis=None,
            rescale_sigma=True,
            mode=params["denoise_wavelet_mode"],
        )
        applied_ops.append("Wavelet denoise (pre)")

    # Step 2: Mild CLAHE for contrast
    needs_contrast = any(
        issue in issues for issue in ("low_contrast", "clipping_low", "clipping_high")
    )
    if needs_contrast:
        kernel_size = params["clahe_tile_size"]
        enhanced = exposure.equalize_adapthist(
            enhanced,
            clip_limit=params["clahe_clip_limit"],
            kernel_size=kernel_size,
        )
        applied_ops.append(
            f"CLAHE (clip={params['clahe_clip_limit']}, tile={kernel_size})"
        )

    # Step 3: Subtle gamma correction
    if "clipping_low" in issues and "clipping_high" not in issues:
        enhanced = exposure.adjust_gamma(enhanced, gamma=params["gamma_brighten"])
        applied_ops.append(f"Gamma brighten ({params['gamma_brighten']})")
    elif "clipping_high" in issues and "clipping_low" not in issues:
        enhanced = exposure.adjust_gamma(enhanced, gamma=params["gamma_darken"])
        applied_ops.append(f"Gamma darken ({params['gamma_darken']})")

    # Step 4: Gentle unsharp mask
    if "blur" in issues:
        enhanced = filters.unsharp_mask(
            enhanced,
            radius=params["unsharp_radius"],
            amount=params["unsharp_amount"],
        )
        applied_ops.append(
            f"Unsharp mask (r={params['unsharp_radius']}, a={params['unsharp_amount']})"
        )

    # Step 5: Light post-denoise
    if "blur" in issues and params["post_denoise_strength"] > 0:
        enhanced = _light_denoise(enhanced, strength=params["post_denoise_strength"])
        applied_ops.append(
            f"Light denoise (post, s={params['post_denoise_strength']})"
        )

    enhanced = np.clip(enhanced, 0.0, 1.0)

    # Safeguard: auto-fix noise amplification
    if _check_noise_amplification(image, enhanced):
        logger.warning("Noise amplification detected — applying corrective denoise.")
        enhanced = _light_denoise(enhanced, strength=0.4)
        applied_ops.append("Auto-corrective denoise (noise guard)")
        enhanced = np.clip(enhanced, 0.0, 1.0)

    return enhanced.astype(np.float32), applied_ops


# ---------------------------------------------------------------------------
# Parameterised enhancement (GenAI-driven)
# ---------------------------------------------------------------------------


def apply_enhancements_from_params(
    image: np.ndarray, plan: "EnhancementPlan"
) -> Tuple[np.ndarray, List[str]]:
    """Apply enhancements using LLM-chosen parameters with safety clamping.

    Same 5-step pipeline as ``apply_enhancements`` but extended with optional
    bilateral and TV-denoise steps and parameterised by an ``EnhancementPlan``.
    """
    from pipeline.schemas import PARAM_BOUNDS

    p = plan.params
    ops = [op.lower().strip() for op in plan.recommended_ops]

    # ---- Safety clamps ----
    def _clamp(val: float, key: str) -> float:
        lo, hi = PARAM_BOUNDS.get(key, (val, val))
        return max(lo, min(hi, val))

    clip_limit = _clamp(p.clahe_clip_limit, "clahe_clip_limit")
    tile_size = int(_clamp(p.clahe_tile_size, "clahe_tile_size"))
    gamma = _clamp(p.gamma, "gamma")
    u_radius = _clamp(p.unsharp_radius, "unsharp_radius")
    u_amount = _clamp(p.unsharp_amount, "unsharp_amount")
    dn_mode = p.denoise_mode if p.denoise_mode in ("soft", "hard") else "soft"
    post_str = _clamp(p.post_denoise_strength, "post_denoise_strength")
    bilateral_d = int(_clamp(p.bilateral_d, "bilateral_d"))
    bilateral_sc = _clamp(p.bilateral_sigma_color, "bilateral_sigma_color")
    bilateral_ss = _clamp(p.bilateral_sigma_space, "bilateral_sigma_space")
    tv_weight = _clamp(p.tv_denoise_weight, "tv_denoise_weight")

    enhanced = image.copy()
    applied_ops: List[str] = []

    # Step 1: Pre-denoise
    if "denoise" in ops:
        enhanced = denoise_wavelet(
            enhanced, channel_axis=None, rescale_sigma=True, mode=dn_mode,
        )
        applied_ops.append(f"Wavelet denoise (pre, mode={dn_mode})")

    # Step 2: CLAHE
    if "clahe" in ops:
        enhanced = exposure.equalize_adapthist(
            enhanced, clip_limit=clip_limit, kernel_size=tile_size,
        )
        applied_ops.append(f"CLAHE (clip={clip_limit:.4f}, tile={tile_size})")

    # Step 3: Gamma correction
    if "gamma" in ops and abs(gamma - 1.0) > 1e-4:
        enhanced = exposure.adjust_gamma(enhanced, gamma=gamma)
        label = "brighten" if gamma < 1.0 else "darken"
        applied_ops.append(f"Gamma {label} ({gamma:.3f})")

    # Step 4: Unsharp mask
    if "unsharp" in ops:
        enhanced = filters.unsharp_mask(
            enhanced, radius=u_radius, amount=u_amount,
        )
        applied_ops.append(f"Unsharp mask (r={u_radius:.2f}, a={u_amount:.2f})")

    # Step 5: Post-denoise
    if "post_denoise" in ops and post_str > 0:
        enhanced = _light_denoise(enhanced, strength=post_str)
        applied_ops.append(f"Light denoise (post, s={post_str:.2f})")

    # Step 6 (optional): Bilateral filter
    if "bilateral" in ops and bilateral_d > 0:
        enhanced = _bilateral_filter(
            enhanced, d=bilateral_d, sigma_color=bilateral_sc, sigma_space=bilateral_ss,
        )
        applied_ops.append(
            f"Bilateral (d={bilateral_d}, sc={bilateral_sc:.3f}, ss={bilateral_ss:.3f})"
        )

    # Step 7 (optional): Total Variation denoise
    if "tv_denoise" in ops and tv_weight > 0:
        enhanced = denoise_tv_chambolle(enhanced, weight=tv_weight, channel_axis=None)
        applied_ops.append(f"TV denoise (w={tv_weight:.4f})")

    enhanced = np.clip(enhanced, 0.0, 1.0)

    # --- Safeguards ---

    # Halo check: if detected, reduce sharpening and re-apply
    if "unsharp" in ops and _check_halo(enhanced):
        logger.warning(
            "Halo detected (edge_ratio > 1.5) — re-applying with halved unsharp_amount."
        )
        enhanced = image.copy()
        reduced_amount = u_amount * 0.5
        # Re-run the full pipeline with reduced sharpening
        for op in ops:
            if op == "denoise":
                enhanced = denoise_wavelet(
                    enhanced, channel_axis=None, rescale_sigma=True, mode=dn_mode,
                )
            elif op == "clahe":
                enhanced = exposure.equalize_adapthist(
                    enhanced, clip_limit=clip_limit, kernel_size=tile_size,
                )
            elif op == "gamma" and abs(gamma - 1.0) > 1e-4:
                enhanced = exposure.adjust_gamma(enhanced, gamma=gamma)
            elif op == "unsharp":
                enhanced = filters.unsharp_mask(
                    enhanced, radius=u_radius, amount=reduced_amount,
                )
            elif op == "post_denoise" and post_str > 0:
                enhanced = _light_denoise(enhanced, strength=post_str)
            elif op == "bilateral" and bilateral_d > 0:
                enhanced = _bilateral_filter(
                    enhanced, d=bilateral_d,
                    sigma_color=bilateral_sc, sigma_space=bilateral_ss,
                )
            elif op == "tv_denoise" and tv_weight > 0:
                enhanced = denoise_tv_chambolle(
                    enhanced, weight=tv_weight, channel_axis=None,
                )
        enhanced = np.clip(enhanced, 0.0, 1.0)
        applied_ops.append(f"[safeguard] Unsharp reduced to {reduced_amount:.2f}")

    # Noise amplification guard
    if _check_noise_amplification(image, enhanced):
        logger.warning("Noise amplification detected — applying corrective denoise.")
        enhanced = _light_denoise(enhanced, strength=0.4)
        applied_ops.append("Auto-corrective denoise (noise guard)")
        enhanced = np.clip(enhanced, 0.0, 1.0)

    # Over-processing guard
    if _check_over_processing(image, enhanced, max_niqe_degradation=0.5):
        logger.warning("Over-processing detected (NIQE degraded >0.5). Blending back.")
        enhanced = 0.6 * enhanced + 0.4 * image
        applied_ops.append("Blend-back 40% original (over-processing guard)")
        enhanced = np.clip(enhanced, 0.0, 1.0)

    return enhanced.astype(np.float32), applied_ops
