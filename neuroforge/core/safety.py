"""
neuroforge/core/safety.py
==========================
Hardware safety invariants for CANRP-X v3.0.

Two critical guards:
  1. Shannon Charge-Density Safety — enforces k = log₁₀(D) + log₁₀(A) ≤ 1.75
     where D = charge density (μC/cm²), A = phase amplitude (μA).
  2. Fall / Trip Reflex Interlock — triggers 85 Nm co-contraction torque when
     IMU angular velocity exceeds 300 °/s.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Tuple

from neuroforge.core.config import (
    SHANNON_K_LIMIT,
    SAFE_CURRENT_AMPLITUDE_UA,
    IMU_TRIP_THRESHOLD_DPS,
    BRACE_TORQUE_NM,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shannon Safety
# ---------------------------------------------------------------------------
@dataclass
class ShannonSafetyResult:
    is_safe: bool
    k_value: float
    clamped_amplitude_ua: float
    charge_density_uc_cm2: float
    alert_message: str


def shannon_safety_check(
    amplitude_ua: float,
    pulse_width_us: float,
    electrode_area_cm2: float = 0.001,
) -> ShannonSafetyResult:
    """
    Enforce the Shannon charge-density safety equation:
        k = log₁₀(D) + log₁₀(A)  must be ≤ 1.75

    Args:
        amplitude_ua      : Stimulation amplitude (μA).
        pulse_width_us    : Phase duration (μs).
        electrode_area_cm2: Active electrode surface area (cm²).

    Returns:
        ShannonSafetyResult with clamped amplitude if violated.
    """
    # Charge per phase (μC)
    charge_uc = amplitude_ua * (pulse_width_us * 1e-6)
    # Charge density (μC / cm²)
    D = charge_uc / electrode_area_cm2
    A = amplitude_ua

    if D <= 0 or A <= 0:
        return ShannonSafetyResult(
            is_safe=False,
            k_value=float("-inf"),
            clamped_amplitude_ua=0.0,
            charge_density_uc_cm2=D,
            alert_message="Non-positive charge density or amplitude.",
        )

    k = math.log10(D) + math.log10(A)
    is_safe = k < SHANNON_K_LIMIT
    clamped = amplitude_ua

    msg = f"k={k:.3f} (A={amplitude_ua:.1f}μA, D={D:.3f}μC/cm²)"

    if not is_safe:
        # Back-solve: max A such that log₁₀(D') + log₁₀(A) = 1.75
        # D' = A * pw / area  → log₁₀(A² * pw/area) ≤ 1.75
        # 2*log₁₀(A) + log₁₀(pw/area) ≤ 1.75
        log_ratio = math.log10(pulse_width_us * 1e-6 / electrode_area_cm2)
        log_a_max = (SHANNON_K_LIMIT - log_ratio) / 2.0
        clamped = min(10 ** log_a_max, SAFE_CURRENT_AMPLITUDE_UA)
        msg = (
            f"SAFETY VIOLATION — k={k:.3f} ≥ {SHANNON_K_LIMIT}. "
            f"Clamping {amplitude_ua:.1f}μA → {clamped:.1f}μA."
        )
        log.warning(msg)

    return ShannonSafetyResult(
        is_safe=is_safe,
        k_value=k,
        clamped_amplitude_ua=clamped,
        charge_density_uc_cm2=D,
        alert_message=msg,
    )


# ---------------------------------------------------------------------------
# Fall / Trip Reflex Interlock
# ---------------------------------------------------------------------------
@dataclass
class ReflexGuardEvent:
    triggered: bool
    angular_velocity_dps: float
    brace_torque_nm: float
    latency_us: float       # simulated latency in microseconds
    timestamp: float


def fall_trip_reflex_guard(
    imu_angular_velocity_dps: float,
    foot_contact_force_n: float = 0.0,
) -> ReflexGuardEvent:
    """
    Microsecond fall/trip reflex interlock.

    Trigger condition:
      - |ω| > 300 °/s  AND  foot-contact force ≈ 0 (aerial phase).

    When triggered:
      - Issues 85 Nm co-contraction bracing torque.
      - Overrides high-level cortical intent (returned flag).
      - Target latency: < 250 μs (emulated here; real implementation on RISC-V NPU).

    Args:
        imu_angular_velocity_dps: Magnitude of IMU angular rate (°/s).
        foot_contact_force_n    : Ground-reaction force estimate (N).

    Returns:
        ReflexGuardEvent dataclass.
    """
    t0 = time.perf_counter()

    aerial_phase = foot_contact_force_n < 5.0   # N threshold for "in-air"
    triggered = (
        abs(imu_angular_velocity_dps) > IMU_TRIP_THRESHOLD_DPS
        and aerial_phase
    )

    torque = BRACE_TORQUE_NM if triggered else 0.0
    latency_us = (time.perf_counter() - t0) * 1e6   # μs

    if triggered:
        log.warning(
            "⚡ FALL REFLEX TRIGGERED — ω=%.1f°/s, bracing torque=%.0fNm, "
            "latency=%.1fμs",
            imu_angular_velocity_dps,
            torque,
            latency_us,
        )

    return ReflexGuardEvent(
        triggered=triggered,
        angular_velocity_dps=imu_angular_velocity_dps,
        brace_torque_nm=torque,
        latency_us=latency_us,
        timestamp=time.time(),
    )
