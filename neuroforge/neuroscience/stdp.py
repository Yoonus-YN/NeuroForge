"""
neuroforge/neuroscience/stdp.py
================================
Triplet Spike-Timing-Dependent Plasticity (STDP) for activity-dependent
corticospinal pathway remodelling during sleep-staged PAS sessions.

Triplet STDP rule (Pfister & Gerstner 2006):
  Δw⁺ = exp(−Δt₁/τ₊) · [A₂⁺ + A₃⁺ · exp(−Δt₂/τ_y)]
  Δw⁻ = exp(−Δt₁/τ₋) · [A₂⁻ + A₃⁻ · exp(−Δt₃/τ_x)]

where:
  Δt₁ = t_post − t_pre
  Δt₂ = t_post − t_post_previous  (inter-postsynaptic interval)
  Δt₃ = t_pre  − t_pre_previous   (inter-presynaptic interval)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from neuroforge.core.config import (
    STDP_A2_PLUS, STDP_A3_PLUS, STDP_TAU_PLUS, STDP_TAU_Y,
    STDP_A2_MINUS, STDP_TAU_MINUS,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default triplet STDP parameters
# ---------------------------------------------------------------------------
A3_MINUS: float = 7.5e-3
TAU_X: float = 101e-3   # s — pre-synaptic eligibility trace (visual cortex fit)


@dataclass
class SynapticState:
    """Per-synapse weight and eligibility traces."""
    weight: float = 0.45          # Synaptic efficacy ∈ [0, 1]
    r1: float = 0.0               # Nearest-neighbour pre-trace
    r2: float = 0.0               # All-to-all pre-trace (x)
    o1: float = 0.0               # Nearest-neighbour post-trace
    o2: float = 0.0               # All-to-all post-trace (y)
    potentiation_total: float = 0.0
    depression_total: float = 0.0
    update_count: int = 0


def triplet_stdp_update(
    state: SynapticState,
    pre_spike: bool,
    post_spike: bool,
    dt: float,
    w_min: float = 0.0,
    w_max: float = 1.0,
    a2_plus: float = STDP_A2_PLUS,
    a3_plus: float = STDP_A3_PLUS,
    tau_plus: float = STDP_TAU_PLUS,
    tau_y: float = STDP_TAU_Y,
    a2_minus: float = STDP_A2_MINUS,
    a3_minus: float = A3_MINUS,
    tau_minus: float = STDP_TAU_MINUS,
    tau_x: float = TAU_X,
) -> SynapticState:
    """
    Apply one time-step of triplet STDP.

    Args:
        state     : Current SynapticState (mutated and returned).
        pre_spike : Boolean — pre-synaptic neuron fired this step.
        post_spike: Boolean — post-synaptic neuron fired this step.
        dt        : Time step (s).

    Returns:
        Updated SynapticState.
    """
    # Exponential decay of traces
    state.r1 *= np.exp(-dt / tau_plus)
    state.r2 *= np.exp(-dt / tau_x)
    state.o1 *= np.exp(-dt / tau_minus)
    state.o2 *= np.exp(-dt / tau_y)

    dw = 0.0

    # Post-synaptic spike: potentiation
    if post_spike:
        dw_plus = state.r1 * (a2_plus + a3_plus * state.o2)
        dw += dw_plus
        state.potentiation_total += dw_plus
        state.o1 += 1.0
        state.o2 += 1.0

    # Pre-synaptic spike: depression
    if pre_spike:
        dw_minus = -state.o1 * (a2_minus + a3_minus * state.r2)
        dw += dw_minus
        state.depression_total += abs(dw_minus)
        state.r1 += 1.0
        state.r2 += 1.0

    state.weight = float(np.clip(state.weight + dw, w_min, w_max))
    state.update_count += 1
    return state


def simulate_pas_session(
    n_cycles: int = 180,
    n_synapses: int = 50,
    dt: float = 1e-3,
    pre_freq_hz: float = 5.0,
    post_freq_hz: float = 10.0,
    jitter_ms: float = 2.0,
    seed: int = 42,
) -> list[float]:
    """
    Simulate a Paired Associative Stimulation (PAS) session over `n_cycles`
    slow-wave sleep-spindle cycles, each containing 2-second bursts.

    Args:
        n_cycles   : Number of NREM spindle cycles (~180 over 6 hours).
        n_synapses : Corticospinal synapses to model.
        dt         : Time step (s).
        pre_freq_hz: Pre-synaptic (cortical) spike frequency (Hz).
        post_freq_hz: Post-synaptic (spinal) spike frequency (Hz).
        jitter_ms  : Spike-timing jitter (ms).

    Returns:
        List of mean synaptic weight after each cycle.
    """
    rng_local = np.random.default_rng(seed)
    synapses = [SynapticState(weight=0.45) for _ in range(n_synapses)]

    burst_dur_s = 2.0
    n_steps = int(burst_dur_s / dt)
    mean_weights: list[float] = []

    for cycle in range(n_cycles):
        # Pre- and post-synaptic spike trains (Poisson)
        pre_spikes = rng_local.random((n_synapses, n_steps)) < pre_freq_hz * dt
        post_spikes = rng_local.random((n_synapses, n_steps)) < post_freq_hz * dt

        for t in range(n_steps):
            for i, syn in enumerate(synapses):
                triplet_stdp_update(
                    syn,
                    pre_spike=bool(pre_spikes[i, t]),
                    post_spike=bool(post_spikes[i, t]),
                    dt=dt,
                )

        mean_w = float(np.mean([s.weight for s in synapses]))
        mean_weights.append(mean_w)

        if (cycle + 1) % 30 == 0:
            log.info(
                "PAS cycle %d/%d — mean synaptic weight: %.3f",
                cycle + 1, n_cycles, mean_w,
            )

    log.info(
        "PAS session complete. Weight: %.3f → %.3f (Δ = +%.3f)",
        0.45, mean_weights[-1], mean_weights[-1] - 0.45,
    )
    return mean_weights
