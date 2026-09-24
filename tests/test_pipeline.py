"""
tests/test_pipeline.py
=======================
Integration smoke-tests for the CANRP-X v3.0 pipeline.
Verifies that every core module initialises and executes without error.
"""
from __future__ import annotations
import numpy as np
import torch
import pytest


# ── Signals ──────────────────────────────────────────────────────────────────
def test_signal_generator():
    from neuroforge.signals.generator import generate_frame
    frame = generate_frame(0)
    assert frame.ecog.shape[0] == 16
    assert 0 < frame.dopamine_nm < 300
    assert 0.5 <= frame.cole_alpha <= 0.95


def test_sqi():
    from neuroforge.signals.generator import generate_frame
    from neuroforge.signals.sqi import compute_sqi
    frame = generate_frame(0, inject_artifact_channel=3)
    result = compute_sqi(frame.ecog)
    assert result.sqi_per_channel.shape == (16,)
    assert 0.0 <= result.mean_sqi <= 1.0


def test_kriging():
    from neuroforge.signals.generator import generate_frame
    from neuroforge.signals.kriging import apply_fault_recovery
    frame = generate_frame(0)
    bad = np.array([2, 7])
    ecog_fixed = apply_fault_recovery(frame.ecog, bad)
    assert ecog_fixed.shape == frame.ecog.shape
    # Imputed channels should be non-zero
    assert np.abs(ecog_fixed[2]).max() > 0


# ── Safety ────────────────────────────────────────────────────────────────────
def test_shannon_safe():
    from neuroforge.core.safety import shannon_safety_check
    # Small amplitude + short pulse → k well below 1.75
    r = shannon_safety_check(10.0, 50.0, electrode_area_cm2=0.01)
    assert r.is_safe, f"Expected safe but k={r.k_value:.3f}: {r.alert_message}"

def test_shannon_violation():
    from neuroforge.core.safety import shannon_safety_check
    r = shannon_safety_check(200.0, 600.0)
    assert not r.is_safe
    assert r.clamped_amplitude_ua < 200.0

def test_fall_reflex_idle():
    from neuroforge.core.safety import fall_trip_reflex_guard
    r = fall_trip_reflex_guard(50.0, foot_contact_force_n=30.0)
    assert not r.triggered

def test_fall_reflex_triggered():
    from neuroforge.core.safety import fall_trip_reflex_guard
    r = fall_trip_reflex_guard(400.0, foot_contact_force_n=0.0)
    assert r.triggered
    assert r.brace_torque_nm == 85.0


# ── Neuroscience ──────────────────────────────────────────────────────────────
def test_cpg_step():
    from neuroforge.neuroscience.cpg import init_cpg_state, step_cpg, get_motor_output
    state = init_cpg_state()
    for _ in range(100):
        state = step_cpg(state)
    out = get_motor_output(state)
    assert 0 <= out["gait_phase_rad"] < 2 * np.pi
    assert out["flexor_rate_hz"] >= 0


def test_attractor():
    from neuroforge.neuroscience.attractor import AttractorState, step_attractor
    state = AttractorState(mode="gait")
    for _ in range(50):
        state = step_attractor(state)
    assert -5.0 < state.x < 5.0


def test_stdp():
    from neuroforge.neuroscience.stdp import SynapticState, triplet_stdp_update
    syn = SynapticState(weight=0.5)
    triplet_stdp_update(syn, pre_spike=True, post_spike=True, dt=1e-3)
    assert 0.0 <= syn.weight <= 1.0


# ── AI ────────────────────────────────────────────────────────────────────────
def test_encoder_encode():
    from neuroforge.ai.encoder import NeuroEncoder
    enc = NeuroEncoder()
    x = torch.randn(2, 16, 200)
    out = enc(x, mode="encode")
    assert "z" in out
    assert out["z"].shape == (2, 128)


def test_encoder_mae():
    from neuroforge.ai.encoder import NeuroEncoder
    enc = NeuroEncoder()
    # n_samples must be divisible into complete 25ms patches (50 samples at 2000 Hz)
    # Use 400 samples = 8 patches
    x = torch.randn(2, 16, 400)
    out = enc(x, mode="mae")
    assert "mae_loss" in out
    assert torch.isfinite(out["mae_loss"])


def test_encoder_simclr():
    from neuroforge.ai.encoder import NeuroEncoder, info_nce_loss
    enc = NeuroEncoder()
    x = torch.randn(4, 16, 200)
    h1 = enc(x, mode="simclr")["h"]
    h2 = enc(x, mode="simclr")["h"]
    loss = info_nce_loss(h1, h2)
    assert loss.item() >= 0


def test_decoder():
    from neuroforge.ai.decoder import EnsembleDecoder
    dec = EnsembleDecoder()
    z = torch.randn(1, 8, 128)
    lidar = torch.rand(1, 8)
    out = dec(z, lidar)
    assert "angles_deg" in out
    assert out["angles_deg"].shape == (1, 3)


def test_domain_adapt_mmd():
    from neuroforge.ai.domain_adapt import mmd_loss
    zs = torch.randn(16, 64)
    zt = torch.randn(16, 64)
    loss = mmd_loss(zs, zt)
    assert loss.item() >= 0


def test_riemannian_align():
    from neuroforge.ai.domain_adapt import riemannian_procrustes_align
    n = 8
    A = np.random.randn(n, n)
    Cs = A @ A.T + np.eye(n) * 0.1
    B = np.random.randn(n, n)
    Ct = B @ B.T + np.eye(n) * 0.1
    X = np.random.randn(100, n)
    Xa = riemannian_procrustes_align(Cs, Ct, X)
    assert Xa.shape == (100, n)


# ── Control ───────────────────────────────────────────────────────────────────
def test_nmpc():
    from neuroforge.control.nmpc import nmpc_solve, default_joint_state
    state, target = default_joint_state(gait_phase=1.0)
    result = nmpc_solve(state, target)
    assert result.optimal_torques.shape == (3,)
    assert result.cost >= 0


# ── XAI ──────────────────────────────────────────────────────────────────────
def test_xai_temporal():
    from neuroforge.xai.attribution import temporal_patch_attribution
    attr = np.random.rand(16, 1000)
    patches = temporal_patch_attribution(attr)
    assert patches.ndim == 1
    assert len(patches) > 0


def test_xai_bands():
    from neuroforge.xai.attribution import frequency_band_attribution
    x = np.random.randn(16, 2000)
    attr = np.random.randn(16, 2000)
    bands = frequency_band_attribution(x, attr)
    assert set(bands.keys()) == {"delta","theta","alpha","mu_beta","low_gamma","high_gamma"}
    assert abs(sum(bands.values()) - 1.0) < 0.01


# ── Digital Twin ──────────────────────────────────────────────────────────────
def test_digital_twin_step():
    from neuroforge.twin.digital_twin import DigitalTwin
    twin = DigitalTwin()
    result = twin.step(0)
    assert result.latency_ms > 0
    assert result.decoded_angles_deg.shape == (3,)
    assert 0.0 <= result.sqi_mean <= 1.0


def test_digital_twin_session():
    from neuroforge.twin.digital_twin import DigitalTwin
    twin = DigitalTwin()
    for i in range(5):
        twin.step(i)
    summary = twin.session_summary()
    assert summary["total_steps"] == 5
    assert summary["mean_latency_ms"] > 0


# ── FHIR ─────────────────────────────────────────────────────────────────────
def test_fhir_export(tmp_path):
    from neuroforge.fhir.telemetry import export_session_bundle, SessionMetrics
    import json
    # Override export dir
    import neuroforge.fhir.telemetry as tel
    tel.FHIR_OUTPUT_DIR = tmp_path
    metrics = SessionMetrics(
        total_steps=50, mean_sqi=0.82, spasticity_episodes=1,
        dopamine_mean_nm=85.0, decoder_latency_ms=9.5,
        synaptic_weight_delta=0.12, session_duration_min=5.0,
    )
    path = export_session_bundle(metrics)
    bundle = json.loads(path.read_text())
    assert bundle["resourceType"] == "Bundle"


# ── Model Card ───────────────────────────────────────────────────────────────
def test_model_card(tmp_path):
    import neuroforge.mlops.model_card as mc
    mc.CARD_OUTPUT_DIR = tmp_path
    path = mc.generate_model_card(
        training_steps=20, mae_loss=0.15, contrastive_loss=1.2,
        mean_sqi=0.82, decoder_latency_ms=9.5,
    )
    import json
    card = json.loads(path.read_text())
    assert card["model_details"]["version"] == "3.0.0"
    assert "audit_trail" in card


# ── Biped Kinematic Simulation ───────────────────────────────────────────────
def test_biped_kinematic_fallback():
    from neuroforge.twin.mujoco_env import KinematicFallback, N_DOF
    sim = KinematicFallback()
    state0 = sim.reset()
    assert state0.joint_angles_rad.shape == (N_DOF,)
    state1 = sim.step(torques=np.zeros(N_DOF), dt=0.01, gait_phase=0.5)
    assert state1.step_count == 1
    assert state1.joint_angles_rad.shape == (N_DOF,)


# ── Continual Learning: EWC ──────────────────────────────────────────────────
def test_ewc_regularizer():
    from neuroforge.ai.decoder import EWCRegularizer
    import torch.nn as nn
    model = nn.Linear(10, 2)
    ewc = EWCRegularizer(model, ewc_lambda=100.0)
    data = [torch.randn(4, 10) for _ in range(3)]
    loss_fn = lambda x: model(x).sum()
    ewc.register_task(data, loss_fn, n_batches=3)
    penalty = ewc.penalty()
    assert float(penalty.detach()) == 0.0  # weights have not moved yet
    with torch.no_grad():
        model.weight += 0.1
    penalty_moved = ewc.penalty()
    assert float(penalty_moved.detach()) > 0.0


# ── Dashboard HTTP Endpoints ────────────────────────────────────────────────
def test_dashboard_handler_endpoints():
    from neuroforge.pipeline import _DashboardHandler, DASHBOARD_DIR, STATE_FILE
    import http.server, threading, urllib.request, time
    import json

    # Ensure a state file exists
    test_state = [{"step": 0, "sqi": 0.95, "latency_ms": 12.0}]
    STATE_FILE.write_text(json.dumps(test_state), encoding="utf-8")
    (DASHBOARD_DIR / "dashboard_state.json").write_text(json.dumps(test_state), encoding="utf-8")

    server = http.server.HTTPServer(("127.0.0.1", 7892), _DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    try:
        r1 = urllib.request.urlopen("http://127.0.0.1:7892/dashboard_state.json")
        assert r1.getcode() == 200
        data1 = json.loads(r1.read())
        assert len(data1) >= 1

        r2 = urllib.request.urlopen("http://127.0.0.1:7892/api/state")
        assert r2.getcode() == 200
        data2 = json.loads(r2.read())
        assert len(data2) >= 1
    finally:
        server.shutdown()

