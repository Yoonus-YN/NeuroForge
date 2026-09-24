"""
neuroforge/xai/attribution.py
================================
Explainable AI attribution suite for CANRP-X v3.0.

Methods:
  1. Path Integrated Gradients — temporal patch and frequency-band relevance.
  2. Frequency-Band Decomposition — isolates per-band model contributions.
  3. Saliency Maps — gradient × input for fast per-patch attribution.

Output flags an XAI alert if low-frequency (artifact-prone) attribution
exceeds high-gamma (physiological) attribution.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sp_signal
from dataclasses import dataclass
from typing import Callable

from neuroforge.core.config import FS_ECOG, N_ECOG_CHANNELS
from neuroforge.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frequency Bands
# ---------------------------------------------------------------------------
BANDS = {
    "delta":    (0.5,  4.0),
    "theta":    (4.0,  8.0),
    "alpha":    (8.0, 12.0),
    "mu_beta": (13.0, 30.0),
    "low_gamma":(30.0, 70.0),
    "high_gamma":(70.0, 150.0),
}


@dataclass
class XAIReport:
    temporal_importance: np.ndarray         # (n_patches,) relevance per patch
    band_importance: dict[str, float]       # per frequency-band attribution
    alert_artifact_dominance: bool          # True if low-freq > high-gamma
    alert_message: str


# ---------------------------------------------------------------------------
# 1. Path Integrated Gradients
# ---------------------------------------------------------------------------
def integrated_gradients(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,                        # (1, n_channels, n_samples)
    target_idx: int = 0,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute path integrated gradients from zero baseline to input x.

        IG(x) = (x − x') · ∫₀¹ (∂F/∂x)|_{x'+α(x−x')} dα

    Args:
        model_fn  : Callable returning scalar output from input tensor.
        x         : Input tensor.
        target_idx: Output dimension to attribute.
        n_steps   : Riemann sum integration steps.

    Returns:
        Attribution map same shape as x.
    """
    x_base = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, n_steps, device=x.device)
    grads = []

    for alpha in alphas:
        x_interp = x_base + alpha * (x - x_base)
        x_interp.requires_grad_(True)
        output = model_fn(x_interp)

        if output.ndim > 0:
            scalar = output[..., target_idx].sum()
        else:
            scalar = output

        scalar.backward()
        grads.append(x_interp.grad.clone())
        x_interp.grad = None

    avg_grad = torch.stack(grads).mean(dim=0)
    ig = (x - x_base) * avg_grad
    return ig.detach()


# ---------------------------------------------------------------------------
# 2. Saliency (gradient × input)
# ---------------------------------------------------------------------------
def saliency_map(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target_idx: int = 0,
) -> torch.Tensor:
    """Gradient × input saliency — faster approximation of IG."""
    x = x.detach().requires_grad_(True)
    # Ensure model parameters don't accumulate gradients during saliency
    output = model_fn(x)
    if output.ndim > 0:
        scalar = output[..., target_idx].sum()
    else:
        scalar = output
    scalar.backward()
    if x.grad is None:
        return torch.zeros_like(x)
    sal = (x * x.grad).detach()
    return torch.nan_to_num(sal.abs(), nan=0.0)


# ---------------------------------------------------------------------------
# 3. Temporal Patch Attribution
# ---------------------------------------------------------------------------
def temporal_patch_attribution(
    attribution: np.ndarray,   # (n_channels, n_samples)
    patch_ms: int = 25,
    fs: float = FS_ECOG,
) -> np.ndarray:
    """
    Pool attribution scores into temporal patches.

    Returns:
        (n_patches,) mean absolute attribution per 25 ms patch.
    """
    patch_samples = int(patch_ms * fs / 1000)
    n_patches = attribution.shape[1] // patch_samples
    patch_attrs = []
    for i in range(n_patches):
        chunk = attribution[:, i * patch_samples:(i + 1) * patch_samples]
        patch_attrs.append(float(np.mean(np.abs(chunk))))
    return np.array(patch_attrs)


# ---------------------------------------------------------------------------
# 4. Frequency-Band Decomposition
# ---------------------------------------------------------------------------
def frequency_band_attribution(
    x_raw: np.ndarray,        # (n_channels, n_samples) — original signal
    attribution: np.ndarray,  # (n_channels, n_samples) — attribution map
    fs: float = FS_ECOG,
) -> dict[str, float]:
    """
    Compute per-band attribution by band-passing both signal and attribution,
    then computing dot-product relevance.

    Returns dict mapping band name → normalised attribution score.
    """
    band_scores: dict[str, float] = {}
    total = 0.0

    for band_name, (lo, hi) in BANDS.items():
        sos = sp_signal.butter(
            4, [lo, min(hi, fs / 2 - 1)], btype="bandpass",
            fs=fs, output="sos",
        )
        x_filt = sp_signal.sosfiltfilt(sos, x_raw, axis=-1)
        a_filt = sp_signal.sosfiltfilt(sos, attribution, axis=-1)
        score = float(np.mean(np.abs(x_filt * a_filt)))
        band_scores[band_name] = score
        total += score

    # Normalise
    if total > 0:
        band_scores = {k: v / total for k, v in band_scores.items()}

    return band_scores


# ---------------------------------------------------------------------------
# XAI Report Generator
# ---------------------------------------------------------------------------
def generate_xai_report(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x_tensor: torch.Tensor,       # (1, n_channels, n_samples)
    x_raw: np.ndarray,            # (n_channels, n_samples) for freq analysis
    target_joint: int = 1,        # knee angle (index 1)
    fs: float = FS_ECOG,
) -> XAIReport:
    """
    Full XAI attribution pipeline:
      1. Compute saliency map (fast approximation).
      2. Extract temporal patch importance.
      3. Extract frequency-band importance.
      4. Check for artifact-dominance alert.

    Returns:
        XAIReport with all attribution fields.
    """
    # Saliency (fast) — use instead of full IG for real-time use
    sal = saliency_map(model_fn, x_tensor, target_idx=target_joint)
    sal_np = sal.squeeze(0).cpu().numpy()    # (n_channels, n_samples)

    temporal_imp = temporal_patch_attribution(sal_np, fs=fs)
    band_imp = frequency_band_attribution(x_raw, sal_np, fs=fs)

    # Alert: artifact dominance if delta+theta > high_gamma
    low_freq_attr = band_imp.get("delta", 0) + band_imp.get("theta", 0)
    high_gamma_attr = band_imp.get("high_gamma", 0)
    alert = low_freq_attr > high_gamma_attr

    msg = (
        f"⚠️  XAI ALERT: Low-frequency artifact dominance detected! "
        f"delta+theta={low_freq_attr:.3f} > high_gamma={high_gamma_attr:.3f}"
        if alert
        else f"✓ XAI OK: high_gamma={high_gamma_attr:.3f}, "
             f"mu_beta={band_imp.get('mu_beta', 0):.3f}"
    )

    if alert:
        log.warning(msg)
    else:
        log.info(msg)

    return XAIReport(
        temporal_importance=temporal_imp,
        band_importance=band_imp,
        alert_artifact_dominance=alert,
        alert_message=msg,
    )
