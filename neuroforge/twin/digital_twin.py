"""
neuroforge/twin/digital_twin.py
=================================
Digital-twin co-simulation testbed.

Integrates all CANRP-X subsystems into a unified closed-loop step function:
  1. Generate synthetic multimodal sensor frame.
  2. SQI evaluation → Kriging imputation.
  3. Safety interlocks (Shannon + fall-reflex).
  4. Attractor CANN step → population vector decoding.
  5. SimCLR encoder forward pass → latent z.
  6. Ensemble decoder → joint angles.
  7. CPG step → motor unit outputs.
  8. NMPC → joint torques.
  9. XAI attribution.
 10. FHIR telemetry accumulation.

All sub-modules operate on CPU; no external physics engine required for the
basic twin (MuJoCo integration is scaffolded via `twin/mujoco_env.py`).
"""
from __future__ import annotations
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from neuroforge.signals.generator import generate_frame, MultimodalFrame
from neuroforge.signals.sqi import compute_sqi
from neuroforge.signals.kriging import apply_fault_recovery
from neuroforge.core.safety import shannon_safety_check, fall_trip_reflex_guard
from neuroforge.neuroscience.cpg import init_cpg_state, step_cpg, get_motor_output
from neuroforge.neuroscience.attractor import AttractorState, step_attractor, population_vector_decode
from neuroforge.ai.encoder import NeuroEncoder
from neuroforge.ai.decoder import EnsembleDecoder
from neuroforge.control.nmpc import nmpc_solve, default_joint_state
from neuroforge.core.config import N_ECOG_CHANNELS, FS_ECOG, PATCH_SIZE_MS
from neuroforge.core.logger import get_logger

log = get_logger(__name__)

PATCH_SAMPLES = int(PATCH_SIZE_MS * FS_ECOG / 1000)
# Keep 4 seconds of latent history (400 patches at 10 ms stride)
LATENT_HISTORY = 8


@dataclass
class TwinStepResult:
    step: int
    latency_ms: float
    sqi_mean: float
    bad_channels: list[int]
    gait_phase_rad: float
    decoded_angles_deg: np.ndarray      # (3,) hip, knee, ankle
    optimal_torques_nm: np.ndarray      # (3,)
    cpg_output: dict
    shannon_ok: bool
    reflex_triggered: bool
    dopamine_nm: float
    cole_alpha: float
    terrain_stair_detected: bool
    xai_band_importance: Optional[dict] = None
    alert_flags: list[str] = field(default_factory=list)


class DigitalTwin:
    """
    Full closed-loop CANRP-X digital-twin simulation engine.
    """

    def __init__(self, n_ecog_channels: int = N_ECOG_CHANNELS):
        log.info("Initialising NeuroForge Digital Twin …")
        self.n_ch = n_ecog_channels

        # Neural network modules (CPU mode)
        self.encoder = NeuroEncoder(n_channels=n_ecog_channels)
        self.encoder.eval()

        self.decoder = EnsembleDecoder()
        self.decoder.eval()

        # Biophysics state
        self.cpg_state = init_cpg_state()
        self.attractor_state = AttractorState(mode="gait")

        # Latent history buffer for sequence decoding
        self._latent_buf: list[torch.Tensor] = []

        # Session metrics accumulators
        self.total_steps = 0
        self.total_latency_ms = 0.0
        self.spasticity_episodes = 0
        self.dopamine_trace: list[float] = []
        self.alert_log: list[str] = []

        log.info("Digital Twin ready.")

    def step(
        self,
        step_index: int,
        inject_artifact: bool = False,
        run_xai: bool = False,
    ) -> TwinStepResult:
        """
        Execute one closed-loop simulation step.

        Args:
            step_index     : Global simulation step counter.
            inject_artifact: Inject an ECoG artifact for fault-recovery demo.
            run_xai        : Compute XAI attribution (slower — ~10 Hz loop).

        Returns:
            TwinStepResult with all subsystem outputs.
        """
        t_start = time.perf_counter()
        alerts: list[str] = []

        # ── 1. Sensor frame ──────────────────────────────────────────────────
        artifact_ch = (step_index % N_ECOG_CHANNELS) if inject_artifact else None
        frame: MultimodalFrame = generate_frame(
            step_index,
            duration_s=0.1,   # 100 ms epoch per step
            inject_motion=True,
            inject_artifact_channel=artifact_ch,
        )

        # ── 2. SQI + Kriging ────────────────────────────────────────────────
        sqi_result = compute_sqi(frame.ecog, fs=float(FS_ECOG))
        ecog_clean = apply_fault_recovery(frame.ecog, sqi_result.bad_channels)

        # ── 3. Safety interlocks ─────────────────────────────────────────────
        # Shannon check on a representative stimulation pulse
        shannon = shannon_safety_check(
            amplitude_ua=80.0,
            pulse_width_us=200.0,
            electrode_area_cm2=0.001,
        )
        if not shannon.is_safe:
            alerts.append(f"SHANNON_VIOLATION: {shannon.alert_message}")

        # Fall-reflex guard
        imu_mag = float(np.linalg.norm(frame.imu_shank_dps))
        reflex = fall_trip_reflex_guard(imu_mag, foot_contact_force_n=20.0)
        if reflex.triggered:
            alerts.append(f"FALL_REFLEX: ω={imu_mag:.0f}°/s")
            self.spasticity_episodes += 1

        # ── 4. Cole-Cole biofouling ──────────────────────────────────────────
        if frame.cole_alpha < 0.70:
            alerts.append(f"COLE_COLE_BIOFOULING: α={frame.cole_alpha:.3f}")

        # ── 5. Terrain detection ─────────────────────────────────────────────
        min_lidar = float(np.min(frame.lidar_distances_m))
        stair_detected = min_lidar < 0.50  # <50 cm obstacle → stair

        # ── 6. CANN attractor step ───────────────────────────────────────────
        u_cortical = np.array([
            0.1 * np.sin(step_index * 0.3),
            0.1 * np.cos(step_index * 0.3),
            0.0,
        ])
        self.attractor_state = step_attractor(
            self.attractor_state, u=u_cortical, dt=0.01,
            switch_to_stance=(step_index == 30),
            switch_to_gait=(step_index == 45),
        )

        # Population vector decoding (estimate firing rates from ECoG RMS)
        rates = np.sqrt(np.mean(ecog_clean ** 2, axis=1))   # (N_ch,) RMS proxy
        pop_vec = population_vector_decode(rates)

        # ── 7. Encoder → latent embedding ───────────────────────────────────
        ecog_tensor = torch.FloatTensor(ecog_clean).unsqueeze(0)  # (1, ch, t)

        with torch.no_grad():
            enc_out = self.encoder(ecog_tensor, mode="encode")
        z = enc_out["z"]   # (1, D_MODEL)

        # Maintain latent history buffer
        self._latent_buf.append(z)
        if len(self._latent_buf) > LATENT_HISTORY:
            self._latent_buf.pop(0)

        # ── 8. Ensemble decoder ──────────────────────────────────────────────
        z_seq = torch.cat(self._latent_buf, dim=0).unsqueeze(0)   # (1, T, D)
        lidar_tensor = torch.FloatTensor(frame.lidar_distances_m).unsqueeze(0)

        with torch.no_grad():
            dec_out = self.decoder(z_seq, lidar_tensor)

        angles_deg = dec_out["angles_deg"].squeeze(0).numpy()

        # ── 9. CPG step ──────────────────────────────────────────────────────
        self.cpg_state = step_cpg(
            self.cpg_state,
            I_descending=8.0,
            dopamine_nm=frame.dopamine_nm,
            dt=0.001,
        )
        cpg_out = get_motor_output(self.cpg_state)

        # ── 10. NMPC ─────────────────────────────────────────────────────────
        joint_state, target_angles = default_joint_state(
            gait_phase=self.cpg_state.gait_phase
        )
        nmpc_result = nmpc_solve(joint_state, target_angles)
        if not nmpc_result.stance_safe:
            alerts.append("NMPC_KNEE_BUCKLE_RISK")

        # ── 11. XAI (optional, slower) ───────────────────────────────────────
        xai_bands = None
        if run_xai:
            from neuroforge.xai.attribution import (
                generate_xai_report,
            )

            def model_fn(inp: torch.Tensor) -> torch.Tensor:
                # Note: no torch.no_grad() here — saliency needs gradients wrt inp
                o = self.encoder(inp, mode="encode")
                z_single = o["z"]                          # (1, D_MODEL)
                seq_exp = z_single.unsqueeze(1).expand(1, LATENT_HISTORY, -1).clone()
                d = self.decoder(seq_exp, lidar_tensor.detach())
                return d["angles_rad"]

            xai_report = generate_xai_report(
                model_fn, ecog_tensor, ecog_clean
            )
            xai_bands = xai_report.band_importance
            # Zero out any lingering gradients on encoder params
            self.encoder.zero_grad(set_to_none=True)
            if xai_report.alert_artifact_dominance:
                alerts.append("XAI_ARTIFACT_DOMINANCE")

        # ── Timing ───────────────────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        if latency_ms > 15.0:
            alerts.append(f"LATENCY_EXCEEDED: {latency_ms:.1f}ms > 15ms")

        self.total_steps += 1
        self.total_latency_ms += latency_ms
        self.dopamine_trace.append(frame.dopamine_nm)
        self.alert_log.extend(alerts)

        result = TwinStepResult(
            step=step_index,
            latency_ms=latency_ms,
            sqi_mean=sqi_result.mean_sqi,
            bad_channels=sqi_result.bad_channels.tolist(),
            gait_phase_rad=self.cpg_state.gait_phase,
            decoded_angles_deg=angles_deg,
            optimal_torques_nm=nmpc_result.optimal_torques,
            cpg_output=cpg_out,
            shannon_ok=shannon.is_safe,
            reflex_triggered=reflex.triggered,
            dopamine_nm=frame.dopamine_nm,
            cole_alpha=frame.cole_alpha,
            terrain_stair_detected=stair_detected,
            xai_band_importance=xai_bands,
            alert_flags=alerts,
        )

        log.info(
            "Step %03d | lat=%.1fms | SQI=%.2f | angles=[%.1f°, %.1f°, %.1f°] "
            "| gait=%.2frad | alerts=%s",
            step_index, latency_ms, sqi_result.mean_sqi,
            angles_deg[0], angles_deg[1], angles_deg[2],
            self.cpg_state.gait_phase, alerts or "none",
        )

        return result

    def session_summary(self) -> dict:
        """Return session-level summary metrics."""
        return {
            "total_steps": self.total_steps,
            "mean_latency_ms": self.total_latency_ms / max(self.total_steps, 1),
            "spasticity_episodes": self.spasticity_episodes,
            "mean_dopamine_nm": float(np.mean(self.dopamine_trace)),
            "total_alerts": len(self.alert_log),
            "unique_alerts": list(set(self.alert_log)),
        }
