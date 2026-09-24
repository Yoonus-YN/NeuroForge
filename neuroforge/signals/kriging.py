"""
neuroforge/signals/kriging.py
==============================
Spatial Kriging imputation for missing / compromised μECoG channels.

Uses a Gaussian variogram model parameterised on a planar 4×4 electrode grid
(matching a 16-contact μECoG array at 3 mm pitch).  When SQI flags a channel
as bad, its spatial neighbours' signals are blended via kriging weights λ that
satisfy the unbiasedness constraint Σλ = 1.

Reference formula:
    V̂(x₀) = Σᵢ λᵢ V(xᵢ),   s.t.  Σᵢ λᵢ = 1
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from neuroforge.core.config import N_ECOG_CHANNELS
from neuroforge.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Electrode geometry — 4 × 4 grid, 3 mm pitch (meters)
# ---------------------------------------------------------------------------
_GRID_SIZE = 4
_PITCH_M = 3e-3
_POSITIONS: np.ndarray = np.array(
    [[(_PITCH_M * (i % _GRID_SIZE), _PITCH_M * (i // _GRID_SIZE))
      for i in range(N_ECOG_CHANNELS)]]
).reshape(N_ECOG_CHANNELS, 2)


def _gaussian_variogram(h: np.ndarray, nugget: float = 0.0,
                         sill: float = 1.0, rang: float = 0.01) -> np.ndarray:
    """γ(h) = nugget + sill*(1 - exp(-(h/range)²))"""
    return nugget + sill * (1.0 - np.exp(-(h / rang) ** 2))


def _build_kriging_system(
    good_positions: np.ndarray,    # (K, 2)
    target_position: np.ndarray,   # (2,)
    nugget: float = 1e-4,
    sill: float = 1.0,
    rang: float = 0.01,
) -> np.ndarray:
    """Return kriging weights λ (size K) for ordinary kriging."""
    K = len(good_positions)

    # Variogram matrix C (K+1 × K+1) with Lagrange multiplier row/col
    dists = np.sqrt(
        np.sum((good_positions[:, None] - good_positions[None, :]) ** 2, axis=-1)
    )
    C = _gaussian_variogram(dists, nugget=nugget, sill=sill, rang=rang)
    # Add Lagrange constraint row/col
    C_ext = np.ones((K + 1, K + 1))
    C_ext[:K, :K] = C
    C_ext[K, K] = 0.0

    # Right-hand side
    d_target = np.sqrt(
        np.sum((good_positions - target_position) ** 2, axis=-1)
    )
    rhs = np.ones(K + 1)
    rhs[:K] = _gaussian_variogram(d_target, nugget=nugget, sill=sill, rang=rang)

    try:
        # Use numpy lstsq (tolerant of near-singular matrices)
        sol, _, _, _ = np.linalg.lstsq(C_ext, rhs, rcond=None)
        weights = sol[:K]
    except np.linalg.LinAlgError:
        log.warning("Kriging system singular — falling back to uniform weights.")
        weights = np.ones(K) / K

    return weights


def kriging_impute(
    ecog: np.ndarray,              # (N_ECOG_CHANNELS, n_samples)
    bad_channels: np.ndarray,      # (M,) int indices
    positions: np.ndarray = _POSITIONS,
) -> np.ndarray:
    """
    Impute bad channels using ordinary spatial Kriging from good neighbours.

    Args:
        ecog        : Raw ECoG (channels × samples). Mutated in-place.
        bad_channels: Array of channel indices requiring imputation.
        positions   : (N_ECOG_CHANNELS, 2) electrode coordinates in metres.

    Returns:
        ECoG array with bad channels replaced by kriging estimates.
    """
    if bad_channels.size == 0:
        return ecog

    ecog_out = ecog.copy()
    good_mask = np.ones(len(positions), dtype=bool)
    good_mask[bad_channels] = False
    good_idx = np.where(good_mask)[0]

    if good_idx.size < 2:
        log.error("Fewer than 2 good channels — cannot impute, zeroing bad channels.")
        ecog_out[bad_channels] = 0.0
        return ecog_out

    good_pos = positions[good_idx]

    for bc in bad_channels:
        target_pos = positions[bc]
        weights = _build_kriging_system(good_pos, target_pos)
        # Weighted sum over good-channel time series
        imputed = np.tensordot(weights, ecog[good_idx], axes=([0], [0]))
        ecog_out[bc] = imputed
        log.debug(
            "Channel %d imputed from %d neighbours (max weight: %.3f)",
            bc, good_idx.size, float(np.max(np.abs(weights))),
        )

    return ecog_out


def apply_fault_recovery(
    ecog: np.ndarray,
    bad_channels: np.ndarray,
) -> np.ndarray:
    """
    High-level fault recovery:
      1. Zero-mask bad channels in the original array.
      2. Kriging-impute from spatial neighbours.

    Returns the imputed array.
    """
    ecog_masked = ecog.copy()
    ecog_masked[bad_channels] = 0.0
    return kriging_impute(ecog_masked, bad_channels)
