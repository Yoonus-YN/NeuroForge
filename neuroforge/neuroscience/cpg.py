"""
neuroforge/neuroscience/cpg.py
================================
Izhikevich Spiking Central Pattern Generator (CPG) for spinal rhythm generation.

Architecture:
  • Flexor pool   (N neurons) — chattering/bursting parameter set
  • Extensor pool (N neurons) — chattering/bursting parameter set
  • Mutual reciprocal Ia inhibitory synapses enforce anti-phase coordination.

Izhikevich model (per neuron):
    dv/dt = 0.04 v² + 5v + 140 − u + I_syn + I_descending
    du/dt = a(bv − u)
    if v ≥ 30 mV → v ← c, u ← u + d

Descending drive I_descending is modulated by dopamine gain from FSCV.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from neuroforge.core.config import (
    IZH_A, IZH_B, IZH_C, IZH_D, IZH_V_THRESH, IZH_V_RESET,
    CPG_N_FLEXOR, CPG_N_EXTENSOR, CPG_MUTUAL_INHIBITION,
    DOPAMINE_BASELINE_NM, SEED,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)
rng = np.random.default_rng(SEED + 1)


# ---------------------------------------------------------------------------
# CPG neuron parameter set (chattering burst — matches native interneurons)
# ---------------------------------------------------------------------------
CPG_PARAMS = dict(a=IZH_A, b=IZH_B, c=IZH_C, d=IZH_D)


@dataclass
class CPGState:
    """Full membrane state of the dual half-center oscillator."""
    v_flex: np.ndarray       # (N_FLEXOR,)   membrane potential (mV)
    u_flex: np.ndarray       # (N_FLEXOR,)   recovery variable
    v_ext: np.ndarray        # (N_EXTENSOR,) membrane potential
    u_ext: np.ndarray        # (N_EXTENSOR,) recovery variable
    spikes_flex: np.ndarray  # (N_FLEXOR,)   bool — fired this step
    spikes_ext: np.ndarray   # (N_EXTENSOR,) bool
    gait_phase: float        # continuous phase ∈ [0, 2π)
    step_count: int = 0


def init_cpg_state() -> CPGState:
    """Initialise CPG with random resting potentials ≈ −65 mV."""
    n_f, n_e = CPG_N_FLEXOR, CPG_N_EXTENSOR
    v_f = rng.uniform(-68.0, -60.0, n_f)
    u_f = IZH_B * v_f
    v_e = rng.uniform(-68.0, -60.0, n_e)
    u_e = IZH_B * v_e
    return CPGState(
        v_flex=v_f, u_flex=u_f,
        v_ext=v_e, u_ext=u_e,
        spikes_flex=np.zeros(n_f, dtype=bool),
        spikes_ext=np.zeros(n_e, dtype=bool),
        gait_phase=0.0,
    )


def _izhikevich_step(
    v: np.ndarray,
    u: np.ndarray,
    I_total: np.ndarray,
    dt: float,
    a: float, b: float, c: float, d: float,
    v_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler step of the Izhikevich neuron model.

    Returns:
        (v_new, u_new, spikes) — all shape (N,)
    """
    dv = (0.04 * v ** 2 + 5.0 * v + 140.0 - u + I_total) * dt
    du = a * (b * v - u) * dt
    v_new = v + dv
    u_new = u + du

    spikes = v_new >= v_thresh
    v_new[spikes] = c
    u_new[spikes] += d

    return v_new, u_new, spikes


def step_cpg(
    state: CPGState,
    I_descending: float = 8.0,
    dopamine_nm: float = DOPAMINE_BASELINE_NM,
    dt: float = 1e-3,        # simulation time step (s)
) -> CPGState:
    """
    Advance the CPG half-center oscillator by one time step.

    Args:
        state         : Current CPGState.
        I_descending  : Cortical/supraspinal descending drive amplitude (μA).
        dopamine_nm   : Dopamine concentration (nM) from FSCV — scales drive.
        dt            : Time step (s).

    Returns:
        Updated CPGState.
    """
    a, b, c, d = CPG_PARAMS["a"], CPG_PARAMS["b"], CPG_PARAMS["c"], CPG_PARAMS["d"]

    # Dopamine-gated gain on descending drive (normalised around baseline)
    dopa_gain = dopamine_nm / DOPAMINE_BASELINE_NM
    I_desc = I_descending * np.clip(dopa_gain, 0.5, 2.0)

    # Synaptic inhibition: each pool inhibits the other proportional to its spike rate
    frac_flex = float(np.mean(state.spikes_flex))
    frac_ext = float(np.mean(state.spikes_ext))

    I_inh_on_ext = CPG_MUTUAL_INHIBITION * frac_flex * 10.0    # current (μA)
    I_inh_on_flex = CPG_MUTUAL_INHIBITION * frac_ext * 10.0

    # Small Gaussian noise term (background synaptic noise)
    noise_f = rng.normal(0, 0.5, CPG_N_FLEXOR)
    noise_e = rng.normal(0, 0.5, CPG_N_EXTENSOR)

    I_flex_total = I_desc + I_inh_on_flex + noise_f
    I_ext_total = I_desc + I_inh_on_ext + noise_e

    v_f, u_f, spk_f = _izhikevich_step(
        state.v_flex, state.u_flex, I_flex_total, dt, a, b, c, d, IZH_V_THRESH
    )
    v_e, u_e, spk_e = _izhikevich_step(
        state.v_ext, state.u_ext, I_ext_total, dt, a, b, c, d, IZH_V_THRESH
    )

    # Update gait phase (0 → 2π per stride cycle based on extensor firing)
    phase_increment = 2.0 * np.pi * dt / 1.2  # ~1.2 s per gait cycle
    new_phase = (state.gait_phase + phase_increment) % (2 * np.pi)

    return CPGState(
        v_flex=v_f, u_flex=u_f,
        v_ext=v_e, u_ext=u_e,
        spikes_flex=spk_f,
        spikes_ext=spk_e,
        gait_phase=new_phase,
        step_count=state.step_count + 1,
    )


def get_motor_output(state: CPGState) -> dict[str, float]:
    """
    Derive functional CPG output metrics.

    Returns:
        Dictionary with flexor_rate, extensor_rate, gait_phase,
        and relative phase (anti-phasic coordination quality).
    """
    flex_rate = float(np.mean(state.spikes_flex)) * 1000.0   # spikes/s estimate
    ext_rate = float(np.mean(state.spikes_ext)) * 1000.0

    # Anti-phase index ∈ [0, 1] — 1 = perfect reciprocal inhibition
    anti_phase = 1.0 - abs(flex_rate - ext_rate) / (flex_rate + ext_rate + 1e-6)

    return {
        "flexor_rate_hz": flex_rate,
        "extensor_rate_hz": ext_rate,
        "gait_phase_rad": state.gait_phase,
        "anti_phase_index": anti_phase,
        "step_count": state.step_count,
    }
