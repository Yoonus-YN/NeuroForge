"""
neuroforge/signals/generator.py
================================
Synthetic biomedical signal generator for the CANRP-X digital-twin testbed.

Generates:
  - 16-channel subdural μECoG / LFP: 1/f background, movement-related
    beta desynchronization (13–30 Hz), high-gamma bursts (70–120 Hz),
    50 Hz line-hum corruption.
  - Fast-Scan Cyclic Voltammetry (FSCV): sub-second dopamine/serotonin
    concentration traces (nM).
  - Cole-Cole tissue-impedance dispersion parameter α over time.
  - 6-DoF IMU angular velocities (thigh + shank).
  - Micro-LiDAR distance readings for terrain classification.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from neuroforge.core.config import (
    FS_ECOG, N_ECOG_CHANNELS, SEED,
    DOPAMINE_BASELINE_NM, SEROTONIN_BASELINE_NM,
    COLE_ALPHA_MIN, FS_FSCV,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pink_noise(n_samples: int, n_channels: int) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping."""
    white = rng.standard_normal((n_channels, n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / FS_ECOG)
    freqs[0] = 1.0  # avoid division by zero at DC
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_filter[0] = 0.0  # zero DC component
    spectrum = np.fft.rfft(white, axis=1) * pink_filter
    return np.fft.irfft(spectrum, n=n_samples, axis=1)


def _add_line_hum(signal: np.ndarray, fs: float, freq: float = 50.0,
                  amplitude: float = 0.15) -> np.ndarray:
    """Add sinusoidal line-hum corruption."""
    t = np.arange(signal.shape[-1]) / fs
    hum = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + hum


def _movement_related_desync(
    signal: np.ndarray,
    fs: float,
    n_samples: int,
    onset_frac: float = 0.30,
    beta_channels: list[int] | None = None,
    gamma_channels: list[int] | None = None,
) -> np.ndarray:
    """
    Inject beta-ERD (13–30 Hz desynchronization) and high-gamma burst
    (70–120 Hz) into specific channels to simulate movement intention onset.
    """
    if beta_channels is None:
        beta_channels = list(range(0, 6))
    if gamma_channels is None:
        gamma_channels = list(range(6, 12))

    onset = int(onset_frac * n_samples)
    t = np.arange(n_samples) / fs

    # Beta desynchronization window (attenuate existing beta)
    beta_carrier = np.sin(2 * np.pi * 20.0 * t)   # 20 Hz representative
    envelope = np.zeros(n_samples)
    envelope[onset:] = np.linspace(0, 1, n_samples - onset)

    for ch in beta_channels:
        signal[ch] -= 0.40 * envelope * beta_carrier   # ERD: amplitude drops

    # High-gamma burst onset
    gamma_carrier = np.sin(2 * np.pi * 90.0 * t)
    gamma_burst = np.zeros(n_samples)
    gamma_burst[onset : onset + int(0.10 * n_samples)] = 1.0

    for ch in gamma_channels:
        signal[ch] += 0.35 * gamma_burst * gamma_carrier

    return signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class MultimodalFrame:
    """One synchronized frame of all sensor modalities."""
    ecog: np.ndarray               # (N_ECOG_CHANNELS, n_samples)
    dopamine_nm: float
    serotonin_nm: float
    cole_alpha: float
    imu_thigh_dps: np.ndarray      # (3,) — roll, pitch, yaw rates
    imu_shank_dps: np.ndarray      # (3,)
    lidar_distances_m: np.ndarray  # (8,) — forward scan sectors
    step_index: int


def generate_frame(
    step_index: int,
    duration_s: float = 0.5,
    inject_motion: bool = True,
    inject_artifact_channel: int | None = None,
) -> MultimodalFrame:
    """
    Generate a synthetic multimodal sensing frame.

    Args:
        step_index           : Simulation step counter (drives time-varying signals).
        duration_s           : Duration of the ECoG epoch (seconds).
        inject_motion        : Whether to inject beta-ERD + gamma burst.
        inject_artifact_channel: Channel index to corrupt (None = no artifact).

    Returns:
        MultimodalFrame with all sensor modalities populated.
    """
    n_samples = int(duration_s * FS_ECOG)

    # ── μECoG / LFP ─────────────────────────────────────────────────────────
    ecog = _pink_noise(n_samples, N_ECOG_CHANNELS)
    ecog = _add_line_hum(ecog, FS_ECOG, freq=50.0, amplitude=0.12)

    if inject_motion:
        ecog = _movement_related_desync(ecog, FS_ECOG, n_samples)

    # Artifact injection (simulate electrode pop / high-impedance)
    if inject_artifact_channel is not None:
        ch = int(inject_artifact_channel) % N_ECOG_CHANNELS
        ecog[ch] += rng.standard_normal(n_samples) * 5.0  # high-amplitude noise
        log.debug("Artifact injected on channel %d", ch)

    # Scale to realistic μV range (roughly ±200 μV peak)
    ecog /= (np.std(ecog, axis=1, keepdims=True) + 1e-9)
    ecog *= 50.0   # μV

    # ── FSCV Dopamine / Serotonin ────────────────────────────────────────────
    phase = step_index * 0.1
    dopamine_nm = (
        DOPAMINE_BASELINE_NM
        + 25.0 * np.sin(phase)
        + rng.normal(0.0, 3.0)
    )
    serotonin_nm = (
        SEROTONIN_BASELINE_NM
        + 8.0 * np.cos(phase * 0.7)
        + rng.normal(0.0, 1.5)
    )

    # ── Cole-Cole α ─────────────────────────────────────────────────────────
    # Slow monotonic decrease simulating glial encapsulation over chronic use
    alpha_base = 0.85 - step_index * 0.0005
    cole_alpha = float(np.clip(alpha_base + rng.normal(0, 0.02), 0.50, 0.95))

    # ── IMU (thigh + shank) ──────────────────────────────────────────────────
    gait_phase = step_index * 0.3
    imu_thigh_dps = np.array([
        40.0 * np.sin(gait_phase),         # roll
        80.0 * np.cos(gait_phase * 1.1),   # pitch (dominant swing axis)
        5.0 * np.sin(gait_phase * 0.3),    # yaw
    ]) + rng.normal(0, 2.0, 3)

    imu_shank_dps = np.array([
        20.0 * np.sin(gait_phase + 0.5),
        120.0 * np.cos(gait_phase * 1.1 + 0.3),
        3.0 * np.sin(gait_phase * 0.3 + 0.1),
    ]) + rng.normal(0, 2.0, 3)

    # ── Micro-LiDAR ─────────────────────────────────────────────────────────
    # 8 angular sectors; inject a stair obstacle at step > 20
    lidar = rng.uniform(0.5, 4.0, size=8)
    if step_index > 20:
        lidar[3] = 0.45   # stair ~45 cm ahead in sector 3 (forward-center)
        lidar[4] = 0.48

    return MultimodalFrame(
        ecog=ecog,
        dopamine_nm=float(np.clip(dopamine_nm, 0, 300)),
        serotonin_nm=float(np.clip(serotonin_nm, 0, 100)),
        cole_alpha=cole_alpha,
        imu_thigh_dps=imu_thigh_dps,
        imu_shank_dps=imu_shank_dps,
        lidar_distances_m=lidar,
        step_index=step_index,
    )
