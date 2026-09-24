"""
neuroforge/twin/mujoco_env.py
==============================
MuJoCo 14-DoF virtual biped environment scaffold.

This module provides the interface for the NeuroForge digital twin to drive
a MuJoCo physics simulation of a 14-degree-of-freedom lower-limb musculoskeletal
model. When MuJoCo is not installed, a lightweight kinematic-only fallback
is used (zero external dependencies beyond NumPy).

Full MuJoCo integration requires:
    pip install mujoco dm-control

The MJCF model file (humanoid_lower_limb.xml) should be placed in:
    neuroforge/twin/assets/humanoid_lower_limb.xml
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from neuroforge.core.logger import get_logger

log = get_logger(__name__)

# Check MuJoCo availability
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    log.info("MuJoCo not installed — using kinematic fallback simulation.")


# ---------------------------------------------------------------------------
# Joint Definitions (14 DoF lower-limb model)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    # Right leg (7 DoF)
    "r_hip_flex", "r_hip_abd", "r_hip_rot",
    "r_knee_flex",
    "r_ankle_flex", "r_ankle_inv", "r_ankle_rot",
    # Left leg (7 DoF)
    "l_hip_flex", "l_hip_abd", "l_hip_rot",
    "l_knee_flex",
    "l_ankle_flex", "l_ankle_inv", "l_ankle_rot",
]

N_DOF = len(JOINT_NAMES)   # 14


@dataclass
class BipedState:
    """Full 14-DoF biped state."""
    joint_angles_rad: np.ndarray     # (14,)
    joint_velocities: np.ndarray     # (14,) rad/s
    joint_torques: np.ndarray        # (14,) Nm
    foot_contact_l: float            # N — left foot GRF
    foot_contact_r: float            # N — right foot GRF
    com_height_m: float              # Centre of mass height (m)
    step_count: int = 0


class KinematicFallback:
    """
    CPU-only kinematic model for the 14-DoF biped.
    Uses sinusoidal gait templates scaled by decoded joint angles.
    No physics — suitable for visual digital-twin output and control testing.
    """

    def __init__(self):
        self.t = 0.0
        self._state = BipedState(
            joint_angles_rad=np.zeros(N_DOF),
            joint_velocities=np.zeros(N_DOF),
            joint_torques=np.zeros(N_DOF),
            foot_contact_l=500.0,
            foot_contact_r=500.0,
            com_height_m=0.95,
        )

    def reset(self) -> BipedState:
        self._state = BipedState(
            joint_angles_rad=np.zeros(N_DOF),
            joint_velocities=np.zeros(N_DOF),
            joint_torques=np.zeros(N_DOF),
            foot_contact_l=500.0,
            foot_contact_r=500.0,
            com_height_m=0.95,
        )
        return self._state

    def step(
        self,
        torques: np.ndarray,          # (14,) target torques
        dt: float = 0.01,
        gait_phase: float = 0.0,
    ) -> BipedState:
        """
        Advance kinematic state by dt seconds.
        Angles are updated using a template gait + torque perturbation.
        """
        self.t += dt

        # Template gait (sinusoidal approximation)
        phase_r = gait_phase
        phase_l = gait_phase + np.pi   # contralateral

        template = np.array([
            # Right leg
            np.deg2rad(20 * np.sin(phase_r)),           # r_hip_flex
            np.deg2rad(5 * np.cos(phase_r * 2)),         # r_hip_abd
            np.deg2rad(3 * np.sin(phase_r * 0.5)),       # r_hip_rot
            np.deg2rad(max(0, 35 * np.abs(np.sin(phase_r / 2)))),  # r_knee
            np.deg2rad(12 * np.sin(phase_r + 0.5)),     # r_ankle_flex
            np.deg2rad(2 * np.cos(phase_r)),             # r_ankle_inv
            np.deg2rad(1.5 * np.sin(phase_r)),           # r_ankle_rot
            # Left leg
            np.deg2rad(20 * np.sin(phase_l)),
            np.deg2rad(5 * np.cos(phase_l * 2)),
            np.deg2rad(3 * np.sin(phase_l * 0.5)),
            np.deg2rad(max(0, 35 * np.abs(np.sin(phase_l / 2)))),
            np.deg2rad(12 * np.sin(phase_l + 0.5)),
            np.deg2rad(2 * np.cos(phase_l)),
            np.deg2rad(1.5 * np.sin(phase_l)),
        ])

        # Small torque influence on angles (simplified compliance)
        damping = 0.003
        self._state.joint_angles_rad = template + damping * torques
        self._state.joint_velocities = (
            self._state.joint_angles_rad - template
        ) / dt
        self._state.joint_torques = torques.copy()

        # Ground contact forces (simplified)
        self._state.foot_contact_r = 500.0 * max(0, np.sin(phase_r + np.pi))
        self._state.foot_contact_l = 500.0 * max(0, np.sin(phase_l + np.pi))
        self._state.com_height_m = 0.95 + 0.03 * np.sin(phase_r * 2)
        self._state.step_count += 1

        return self._state


class MuJoCoEnv:
    """
    Full MuJoCo physics environment wrapper.
    Instantiated only if mujoco package is available.
    """

    def __init__(self, model_path: str | None = None):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError(
                "MuJoCo not installed. Install with: pip install mujoco"
            )
        import mujoco
        from pathlib import Path

        if model_path is None:
            model_path = str(
                Path(__file__).parent / "assets" / "humanoid_lower_limb.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        log.info("MuJoCo model loaded: %d DoF", self.model.nq)

    def reset(self) -> BipedState:
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        return self._get_state()

    def step(self, torques: np.ndarray, dt: float = 0.01) -> BipedState:
        import mujoco
        self.data.ctrl[:] = torques[:self.model.nu]
        mujoco.mj_step(self.model, self.data)
        return self._get_state()

    def _get_state(self) -> BipedState:
        return BipedState(
            joint_angles_rad=self.data.qpos[:N_DOF].copy(),
            joint_velocities=self.data.qvel[:N_DOF].copy(),
            joint_torques=self.data.ctrl[:N_DOF].copy(),
            foot_contact_l=0.0,   # would need contact sensor lookup
            foot_contact_r=0.0,
            com_height_m=float(self.data.subtree_com[0, 2]),
        )


def create_environment() -> "KinematicFallback | MuJoCoEnv":
    """Factory — returns MuJoCo env if available, fallback otherwise."""
    if MUJOCO_AVAILABLE:
        try:
            return MuJoCoEnv()
        except Exception as e:
            log.warning("MuJoCo env creation failed (%s) — using fallback.", e)
    return KinematicFallback()
