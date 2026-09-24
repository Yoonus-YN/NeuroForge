"""
neuroforge/signals/sqi.py
==========================
Multi-Criteria Signal Quality Index (SQI) for electrophysiological channels.

Computes per-channel:
  • Line Noise Ratio (LNR)  — ratio of 50/60 Hz power to broadband power
  • High-Frequency Noise Index (HNI) — power above 300 Hz relative to 1–150 Hz
  • Kurtosis divergence term — penalizes high-amplitude artifact bursts

SQI_i = w1*(1-LNR) + w2*(1-HNI) + w3*exp(-|Kurt - 3| / 2)

Channels below SQI_THRESHOLD (0.75) are flagged for Kriging imputation.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis
from dataclasses import dataclass
from typing import Tuple

from neuroforge.core.config import (
    FS_ECOG, SQI_THRESHOLD, SQI_W1, SQI_W2, SQI_W3,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class SQIResult:
    sqi_per_channel: np.ndarray      # (N,) — float in [0, 1]
    lnr_per_channel: np.ndarray      # (N,) — Line Noise Ratio
    hni_per_channel: np.ndarray      # (N,) — High-Freq Noise Index
    kurt_per_channel: np.ndarray     # (N,) — excess kurtosis
    good_channels: np.ndarray        # (N,) bool — True = healthy
    bad_channels: np.ndarray         # (M,) int indices of compromised channels
    mean_sqi: float


def _band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Compute power in [f_lo, f_hi] Hz using Welch's method."""
    nperseg = min(256, len(x))
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    idx = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.trapz(psd[idx], freqs[idx]))


def compute_sqi(
    ecog: np.ndarray,           # (N_channels, n_samples)
    fs: float = FS_ECOG,
    line_freq: float = 50.0,
    line_bw: float = 2.0,
) -> SQIResult:
    """
    Compute multi-criteria SQI for each channel.

    Args:
        ecog      : Raw ECoG array (channels × samples).
        fs        : Sampling rate (Hz).
        line_freq : Power-line frequency (Hz) — 50 or 60.
        line_bw   : Half-bandwidth around line frequency (Hz).

    Returns:
        SQIResult dataclass.
    """
    n_ch = ecog.shape[0]
    lnr_arr = np.zeros(n_ch)
    hni_arr = np.zeros(n_ch)
    kurt_arr = np.zeros(n_ch)

    for i, ch in enumerate(ecog):
        # Broadband power (0.5 – 250 Hz)
        p_broad = _band_power(ch, fs, 0.5, 250.0) + 1e-12

        # LNR — notch band ± 2 Hz around line frequency
        p_line = _band_power(ch, fs, line_freq - line_bw, line_freq + line_bw)
        lnr_arr[i] = min(p_line / p_broad, 1.0)

        # HNI — power above 300 Hz vs. 1–150 Hz
        p_high = _band_power(ch, fs, 300.0, fs / 2 - 1)
        p_physio = _band_power(ch, fs, 1.0, 150.0) + 1e-12
        hni_arr[i] = min(p_high / p_physio, 1.0)

        # Kurtosis (excess): Gaussian noise → 0, artifact spikes → >> 0
        kurt_arr[i] = float(kurtosis(ch, fisher=True))  # excess kurtosis

    sqi = (
        SQI_W1 * (1.0 - lnr_arr)
        + SQI_W2 * (1.0 - hni_arr)
        + SQI_W3 * np.exp(-np.abs(kurt_arr - 0.0) / 2.0)   # Fisher → 0 for Gaussian
    )
    sqi = np.clip(sqi, 0.0, 1.0)

    good = sqi >= SQI_THRESHOLD
    bad = np.where(~good)[0]

    if bad.size > 0:
        log.warning(
            "SQI: %d/%d channels below threshold %.2f — channels: %s",
            bad.size, n_ch, SQI_THRESHOLD, bad.tolist(),
        )

    return SQIResult(
        sqi_per_channel=sqi,
        lnr_per_channel=lnr_arr,
        hni_per_channel=hni_arr,
        kurt_per_channel=kurt_arr,
        good_channels=good,
        bad_channels=bad,
        mean_sqi=float(np.mean(sqi)),
    )
