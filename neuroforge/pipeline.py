"""
neuroforge/pipeline.py
=======================
CANRP-X v3.0 — Master Closed-Loop Pipeline Runner.

Executes the complete pipeline end-to-end:
  1. Pre-training loop (SimCLR + MAE) — abbreviated for demo.
  2. Closed-loop simulation (N steps).
  3. PAS sleep-plasticity session.
  4. XAI report generation.
  5. FHIR telemetry export.
  6. Regulatory model card generation.
  7. HTTP server for the WebGL dashboard.

Run: python -m neuroforge.pipeline
"""
from __future__ import annotations
import argparse
import json
import math
import threading
import time
import http.server
import webbrowser
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from neuroforge.core.config import SEED
from neuroforge.core.logger import get_logger
from neuroforge.ai.encoder import NeuroEncoder, TemporalAugment, info_nce_loss
from neuroforge.neuroscience.stdp import simulate_pas_session
from neuroforge.twin.digital_twin import DigitalTwin
from neuroforge.fhir.telemetry import export_session_bundle, SessionMetrics
from neuroforge.mlops.model_card import generate_model_card
from neuroforge.core.safety import shannon_safety_check

log = get_logger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"
STATE_FILE = Path(__file__).parents[1] / "dashboard_state.json"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Stage 1: Abbreviated Self-Supervised Pretraining
# ---------------------------------------------------------------------------
def pretrain_encoder(
    encoder: NeuroEncoder,
    n_steps: int = 20,
    batch_size: int = 4,
    n_channels: int = 16,
    n_samples: int = 200,
) -> tuple[list[float], list[float]]:
    """Run abbreviated SimCLR + MAE pretraining loop."""
    log.info("Starting self-supervised pretraining (%d steps) …", n_steps)
    augment = TemporalAugment()
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    mae_losses, nce_losses = [], []

    for step in range(n_steps):
        # Synthetic batch
        x = torch.randn(batch_size, n_channels, n_samples)
        x1 = augment(x)
        x2 = augment(x)

        optimizer.zero_grad()

        # MAE objective
        mae_out = encoder(x, mode="mae")
        mae_loss = mae_out["mae_loss"]

        # SimCLR objective
        h1 = encoder(x1, mode="simclr")["h"]
        h2 = encoder(x2, mode="simclr")["h"]
        nce_loss = info_nce_loss(h1, h2)

        total_loss = mae_loss + nce_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        mae_losses.append(float(mae_loss.detach()))
        nce_losses.append(float(nce_loss.detach()))

        if (step + 1) % 5 == 0:
            log.info(
                "  Pretrain step %02d/%02d — MAE=%.4f  NCE=%.4f",
                step + 1, n_steps, mae_losses[-1], nce_losses[-1],
            )

    log.info("Pretraining complete. Final MAE=%.4f  NCE=%.4f",
             mae_losses[-1], nce_losses[-1])
    return mae_losses, nce_losses


# ---------------------------------------------------------------------------
# Dashboard state writer
# ---------------------------------------------------------------------------
def _write_dashboard_state(results: list[dict]) -> None:
    """Write JSON state consumed by the WebGL dashboard via polling."""
    content = json.dumps(results, indent=2)
    STATE_FILE.write_text(content, encoding="utf-8")
    (DASHBOARD_DIR / "dashboard_state.json").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP server for dashboard
# ---------------------------------------------------------------------------
class _DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path in ("", "/"):
            self.path = "/index.html"
            clean_path = "/index.html"

        if clean_path in ("/dashboard_state.json", "/api/state"):
            target = None
            if (DASHBOARD_DIR / "dashboard_state.json").exists():
                target = DASHBOARD_DIR / "dashboard_state.json"
            elif STATE_FILE.exists():
                target = STATE_FILE

            if target and target.exists():
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"[]")
                return

        return super().do_GET()

    def log_message(self, format, *args):
        pass   # suppress access log noise


class _ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True


def _serve_dashboard(port: int = 7890) -> None:
    """Start a simple HTTP server for the WebGL dashboard."""
    try:
        server = _ReusableHTTPServer(("127.0.0.1", port), _DashboardHandler)
        log.info("Dashboard server → http://127.0.0.1:%d", port)
        server.serve_forever()
    except OSError as e:
        log.warning("Could not bind dashboard server to port %d: %s", port, e)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    n_sim_steps: int = 60,
    pretrain_steps: int = 20,
    serve_dashboard: bool = True,
    open_browser: bool = True,
    port: int = 7890,
) -> dict:
    """
    Execute the full CANRP-X v3.0 simulation pipeline.

    Args:
        n_sim_steps     : Number of closed-loop simulation steps.
        pretrain_steps  : Self-supervised pretraining steps.
        serve_dashboard : Start HTTP server for WebGL dashboard.
        open_browser    : Auto-open dashboard in browser.
        port            : Dashboard HTTP port.

    Returns:
        Summary dictionary with all key metrics.
    """
    log.info("=" * 70)
    log.info("  NeuroForge CANRP-X v3.0 — Master Pipeline Starting")
    log.info("=" * 70)

    # Start dashboard server immediately so browser can connect right away
    if serve_dashboard:
        t = threading.Thread(target=_serve_dashboard, args=(port,), daemon=True)
        t.start()
        time.sleep(0.3)
        log.info("Dashboard server live at http://127.0.0.1:%d", port)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/index.html")

    # Stage 1: Pre-training
    encoder = NeuroEncoder()
    mae_hist, nce_hist = pretrain_encoder(encoder, n_steps=pretrain_steps)

    # Stage 2: Digital twin closed-loop simulation
    twin = DigitalTwin()
    twin.encoder = encoder   # use pretrained encoder

    sim_results: list[dict] = []
    log.info("Starting closed-loop simulation (%d steps) …", n_sim_steps)

    for step in range(n_sim_steps):
        inject = (step % 10 == 0)   # periodic artifact injection
        xai = (step % 5 == 0)       # XAI every 5th step
        result = twin.step(step, inject_artifact=inject, run_xai=xai)

        rec = {
            "step": result.step,
            "latency_ms": round(result.latency_ms, 2),
            "sqi": round(result.sqi_mean, 3),
            "bad_channels": result.bad_channels,
            "gait_phase": round(result.gait_phase_rad, 4),
            "angles_deg": [round(float(a), 2) for a in result.decoded_angles_deg],
            "torques_nm": [round(float(t), 2) for t in result.optimal_torques_nm],
            "cpg": result.cpg_output,
            "shannon_ok": result.shannon_ok,
            "reflex": result.reflex_triggered,
            "dopamine_nm": round(result.dopamine_nm, 2),
            "cole_alpha": round(result.cole_alpha, 3),
            "stair": result.terrain_stair_detected,
            "xai_bands": result.xai_band_importance or {},
            "alerts": result.alert_flags,
            # Attractor state for 3D visualisation
            "attractor_x": round(float(twin.attractor_state.x), 4),
            "attractor_y": round(float(twin.attractor_state.y), 4),
            "attractor_z": round(float(twin.attractor_state.z), 4),
            "attractor_mode": twin.attractor_state.mode,
        }
        sim_results.append(rec)

        # Update dashboard state file after each step
        _write_dashboard_state(sim_results)
        time.sleep(0.05)   # 50 ms pacing — 20 Hz for dashboard updates

    session = twin.session_summary()
    log.info("Simulation complete. Session summary: %s", session)

    # Stage 3: PAS sleep-plasticity
    log.info("Running PAS sleep-plasticity simulation …")
    pas_weights = simulate_pas_session(n_cycles=10, seed=SEED)   # abbreviated

    # Stage 4: Shannon safety demo
    log.info("Shannon safety invariant check …")
    s = shannon_safety_check(amplitude_ua=120.0, pulse_width_us=300.0)
    log.info("  %s", s.alert_message)

    # Stage 5: FHIR export
    metrics = SessionMetrics(
        total_steps=session["total_steps"],
        mean_sqi=float(np.mean([r["sqi"] for r in sim_results])),
        spasticity_episodes=session["spasticity_episodes"],
        dopamine_mean_nm=session["mean_dopamine_nm"],
        decoder_latency_ms=session["mean_latency_ms"],
        synaptic_weight_delta=pas_weights[-1] - 0.45,
        session_duration_min=n_sim_steps * 0.1 / 60.0,
        alert_flags=session["unique_alerts"],
    )
    fhir_path = export_session_bundle(metrics)
    log.info("FHIR bundle → %s", fhir_path)

    # Stage 6: Model card
    card_path = generate_model_card(
        training_steps=pretrain_steps,
        mae_loss=mae_hist[-1],
        contrastive_loss=nce_hist[-1],
        mean_sqi=metrics.mean_sqi,
        decoder_latency_ms=metrics.decoder_latency_ms,
        ablation_results={
            "without_ssl_pretraining": float("nan"),       # placeholder for future ablation
            "without_kriging_imputation": float("nan"),
            "without_domain_adaptation": float("nan"),
        },
    )
    log.info("Model card → %s", card_path)

    summary = {
        "session": session,
        "pretrain": {"mae_final": mae_hist[-1], "nce_final": nce_hist[-1]},
        "pas": {"weight_start": 0.45, "weight_end": pas_weights[-1]},
        "fhir": str(fhir_path),
        "model_card": str(card_path),
        "sim_steps": len(sim_results),
    }

    log.info("=" * 70)
    log.info("  Pipeline complete.")
    log.info("  Mean latency: %.2f ms  (target < 15 ms)", session["mean_latency_ms"])
    log.info("  Mean SQI: %.3f", metrics.mean_sqi)
    log.info("  Synaptic weight Δ: %.3f", pas_weights[-1] - 0.45)
    log.info("=" * 70)

    # Keep dashboard server alive if serving
    if serve_dashboard:
        log.info("Dashboard live at http://127.0.0.1:%d — press Ctrl-C to exit.", port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Pipeline interrupted by user.")

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroForge CANRP-X v3.0 — Master Pipeline"
    )
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps")
    parser.add_argument("--pretrain", type=int, default=20, help="Pretraining steps")
    parser.add_argument("--port", type=int, default=7890, help="Dashboard port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    parser.add_argument("--dashboard-only", action="store_true", help="Start dashboard server only")
    args = parser.parse_args()

    if args.dashboard_only:
        log.info("=" * 70)
        log.info("  NeuroForge CANRP-X v3.0 — Dashboard Standalone Server")
        log.info("=" * 70)
        t = threading.Thread(target=_serve_dashboard, args=(args.port,), daemon=True)
        t.start()
        time.sleep(0.3)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}/index.html")
        log.info("Dashboard live at http://127.0.0.1:%d — press Ctrl-C to exit.", args.port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Dashboard stopped.")
    else:
        run_pipeline(
            n_sim_steps=args.steps,
            pretrain_steps=args.pretrain,
            open_browser=not args.no_browser,
            port=args.port,
        )
