"""
neuroforge/neuroscience/attractor.py
=====================================
Continuous Attractor Neural Network (CANN) — 3D Hopf-type manifold.

The CANN models the motor cortex as a low-dimensional dynamical system:
  • Ring attractor → rhythmic gait cycle (limit cycle on torus)
  • Point attractor → static stance / isometric hold

State vector x ∈ ℝ³ evolves via:
    dx/dt = F(x) + B·u + η

where F(x) encodes:
  − Ring limit-cycle dynamics (Hopf oscillator x-y plane)
  − Stable z-axis point attractor (postural holding)
  − Cubic nonlinearity for bounded trajectories

Population vector decoding extracts motor command:
    P⃗(t) = Σ_k  ((r_k(t) − r̄_k) / σ_k)  C⃗_k
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from neuroforge.core.config import SEED
from neuroforge.core.logger import get_logger

log = get_logger(__name__)
rng = np.random.default_rng(SEED + 2)


# ---------------------------------------------------------------------------
# Hopf oscillator parameters
# ---------------------------------------------------------------------------
HOPF_MU: float = 1.0       # limit-cycle radius
HOPF_OMEGA: float = 2 * np.pi / 1.2   # rad/s  (~1.2 s gait cycle)
HOPF_ALPHA: float = 0.8    # convergence rate to limit cycle
Z_ATTRACTOR: float = 0.0   # z point-attractor target
Z_LAMBDA: float = 3.0      # z-axis convergence rate
NOISE_SIGMA: float = 0.02  # state noise amplitude


@dataclass
class AttractorState:
    x: float = 0.0     # cortical manifold dim 1
    y: float = 0.5     # cortical manifold dim 2
    z: float = 0.0     # postural dim
    mode: str = "gait" # "gait" | "stance"


def _hopf_flow(state: AttractorState, u: np.ndarray, dt: float) -> AttractorState:
    """
    Euler integration of Hopf oscillator + z-axis attractor.

    Args:
        state: Current AttractorState.
        u    : External input (3,) — cortical top-down drive.
        dt   : Time step (s).

    Returns:
        New AttractorState.
    """
    x, y, z = state.x, state.y, state.z
    r2 = x ** 2 + y ** 2

    # Hopf limit-cycle dynamics
    dx = (HOPF_ALPHA * (HOPF_MU - r2) * x - HOPF_OMEGA * y) * dt + u[0] * dt
    dy = (HOPF_ALPHA * (HOPF_MU - r2) * y + HOPF_OMEGA * x) * dt + u[1] * dt
    # z-axis stable attractor
    dz = -Z_LAMBDA * (z - Z_ATTRACTOR) * dt + u[2] * dt

    # Additive state noise
    noise = rng.normal(0, NOISE_SIGMA, 3)

    return AttractorState(
        x=x + dx + noise[0],
        y=y + dy + noise[1],
        z=z + dz + noise[2],
        mode=state.mode,
    )


def _stance_flow(state: AttractorState, u: np.ndarray, dt: float) -> AttractorState:
    """Damp oscillations: converge all dims to fixed-point attractor."""
    lam = 5.0
    dx = -lam * state.x * dt + u[0] * dt
    dy = -lam * state.y * dt + u[1] * dt
    dz = -Z_LAMBDA * (state.z - Z_ATTRACTOR) * dt + u[2] * dt
    noise = rng.normal(0, NOISE_SIGMA * 0.3, 3)
    return AttractorState(
        x=state.x + dx + noise[0],
        y=state.y + dy + noise[1],
        z=state.z + dz + noise[2],
        mode="stance",
    )


def step_attractor(
    state: AttractorState,
    u: np.ndarray | None = None,
    dt: float = 0.01,
    switch_to_stance: bool = False,
    switch_to_gait: bool = False,
) -> AttractorState:
    """
    Step the CANN forward by dt.

    Args:
        state           : Current AttractorState.
        u               : External cortical input (3,) — defaults to zero.
        dt              : Time step (s).
        switch_to_stance: Transition to point attractor (standing).
        switch_to_gait  : Transition to ring attractor (walking).

    Returns:
        Updated AttractorState.
    """
    if u is None:
        u = np.zeros(3)

    if switch_to_gait:
        state.mode = "gait"
        log.info("CANN → gait (ring attractor)")
    if switch_to_stance:
        state.mode = "stance"
        log.info("CANN → stance (point attractor)")

    if state.mode == "gait":
        return _hopf_flow(state, u, dt)
    else:
        return _stance_flow(state, u, dt)


# ---------------------------------------------------------------------------
# Population vector decoding
# ---------------------------------------------------------------------------
def build_tuning_vectors(
    n_channels: int = 16,
    dim: int = 3,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build preferred-direction tuning vectors, mean rates, and std devs
    for a population of `n_channels` cortical channels.

    Returns:
        (C, r_bar, sigma) — shapes (n_channels, dim), (n_channels,), (n_channels,)
    """
    rng_l = np.random.default_rng(seed)
    C = rng_l.standard_normal((n_channels, dim))
    C /= np.linalg.norm(C, axis=1, keepdims=True)  # unit vectors
    r_bar = rng_l.uniform(2.0, 10.0, n_channels)   # mean baseline firing (Hz)
    sigma = rng_l.uniform(0.5, 2.0, n_channels)     # tuning width
    return C, r_bar, sigma


_C, _R_BAR, _SIGMA = build_tuning_vectors()


def population_vector_decode(
    rates: np.ndarray,              # (N_channels,) instantaneous firing rates
    C: np.ndarray = _C,
    r_bar: np.ndarray = _R_BAR,
    sigma: np.ndarray = _SIGMA,
) -> np.ndarray:
    """
    Compute the population vector P⃗(t):

        P⃗(t) = Σ_k  ((r_k − r̄_k) / σ_k)  C⃗_k

    Args:
        rates : (N,) vector of instantaneous channel firing rates (Hz).
        C     : (N, 3) tuning vectors.
        r_bar : (N,) baseline mean rates.
        sigma : (N,) tuning standard deviations.

    Returns:
        P⃗ — 3D decoded motor command vector.
    """
    weights = (rates - r_bar) / (sigma + 1e-9)   # (N,)
    P = weights @ C                               # (3,)
    return P
