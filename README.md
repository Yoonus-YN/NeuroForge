# NeuroForge CANRP-X v3.0
## Closed-Loop Autonomous Neuro-Restoration Platform

> **Research Prototype — TRL 1–3 Software Testbed**  
> Not approved for clinical or human use.

---

## Overview

NeuroForge CANRP-X is a unified, bio-digital, closed-loop neuroprosthetic engine designed to restore volitional movement in individuals with complete paralysis (SCI, Brainstem Stroke, ALS). It maps cortical motor intention into phase-locked spinal epidural stimulation while simultaneously executing a full biophysical digital-twin simulation.

## Repository Layout

```
neuroforge/
├── core/           config, logger, safety interlocks
├── signals/        generator, SQI, Kriging imputation
├── neuroscience/   Izhikevich CPG, Triplet STDP, CANN attractor
├── ai/             SimCLR encoder, MAE, ensemble decoder, domain adapt
├── xai/            Integrated Gradients, freq-band attribution
├── control/        NMPC joint torque solver
├── twin/           Digital twin closed-loop engine
├── fhir/           FHIR R4 / HL7 telemetry export
├── mlops/          Automated model card generator
├── dashboard/      WebGL Three.js dual-canvas dashboard
└── pipeline.py     Master pipeline orchestrator
tests/
logs/
fhir_exports/
model_cards/
dashboard_state.json  (live state polled by dashboard)
```

## Quick Start & How to Run

### 1. Installation
Install NeuroForge in editable development mode:
```bash
pip install -e .
```

### 2. How to Test
Execute the test suite (all 26 integration & unit tests):
```bash
# Run complete test suite
pytest -v

# Or run via Python module
python -m pytest tests/ -v
```

### 3. How to Run the Master Pipeline
Run the full closed-loop simulation, pretraining, sleep-plasticity, FHIR export, model card generation, and start the live WebGL dashboard:
```bash
# Standard run (60 simulation steps, 20 pretraining steps, auto-launch browser)
python -m neuroforge.pipeline

# Custom simulation length and pretraining:
python -m neuroforge.pipeline --steps 120 --pretrain 30 --port 7890

# Headless / server mode (do not open browser automatically):
python -m neuroforge.pipeline --steps 60 --pretrain 10 --no-browser
```

Once started, open your web browser to:
**[http://127.0.0.1:7890](http://127.0.0.1:7890)**

---

## How to Use the Modules (Python API)

### 1. Real-Time Signal Conditioning & Fault Recovery
```python
import numpy as np
from neuroforge.signals.generator import generate_frame
from neuroforge.signals.sqi import compute_sqi
from neuroforge.signals.kriging import apply_fault_recovery

# Generate a synthetic frame with an artifact on channel 3
frame = generate_frame(step_index=0, inject_artifact_channel=3)

# Evaluate Signal Quality Index across 16 μECoG channels
sqi_res = compute_sqi(frame.ecog)
print(f"Mean SQI: {sqi_res.mean_sqi:.3f}, Bad channels: {sqi_res.bad_channels}")

# Spatially impute disconnected/noisy electrodes via Ordinary Kriging
cleaned_ecog = apply_fault_recovery(frame.ecog, sqi_res.bad_channels)
```

### 2. Hardware Safety Invariants (Shannon Limit & Fall Reflex)
```python
from neuroforge.core.safety import shannon_safety_check, fall_trip_reflex_guard

# 1. Shannon charge-density invariant: k = log10(D) + log10(A) <= 1.75
safety = shannon_safety_check(amplitude_ua=120.0, pulse_width_us=300.0)
if not safety.is_safe:
    print(f"Safety violation! Clamped amplitude to: {safety.clamped_amplitude_ua:.1f} μA")

# 2. Fall-reflex interlock (angular velocity > 300 deg/s triggers 85 Nm brace)
reflex = fall_trip_reflex_guard(gyro_dps=350.0, foot_contact_force_n=0.0)
if reflex.triggered:
    print(f"Reflex triggered! Brace torque: {reflex.brace_torque_nm} Nm")
```

### 3. Self-Supervised AI Encoder & Ensemble Decoder
```python
import torch
from neuroforge.ai.encoder import NeuroEncoder
from neuroforge.ai.decoder import EnsembleDecoder

encoder = NeuroEncoder()
decoder = EnsembleDecoder()

# Input: (batch, channels, samples) = (1, 16, 200)
x = torch.randn(1, 16, 200)
z = encoder(x, mode="encode")["z"]  # (1, 128) latent vector

# Decoder inputs: sequence of latents + LiDAR terrain vector (1, 8)
z_seq = z.unsqueeze(1).expand(1, 8, 128)
lidar = torch.zeros(1, 8)  # flat terrain
out = decoder(z_seq, lidar)

print(f"Decoded Joint Angles (deg): {out['angles_deg']}")
print(f"Transformer vs SSM weight: {out['weight_transformer']:.2f} / {out['weight_ssm']:.2f}")
```

### 4. Digital Twin Simulation
```python
from neuroforge.twin.digital_twin import DigitalTwin

twin = DigitalTwin()
result = twin.step(step_index=0, inject_artifact=False, run_xai=True)

print(f"Step {result.step} | Latency: {result.latency_ms:.1f} ms | SQI: {result.sqi_mean:.2f}")
print(f"Decoded Angles: {result.decoded_angles_deg}")
```

---

## Architecture Pillars

| Domain | Key Modules |
|--------|-------------|
| AI & Deep Learning | SimCLR, MAE, Patch-Transformer, Mamba SSM, EWC |
| Computational Neuroscience | Izhikevich CPG, Triplet STDP, Hopf CANN, Population Vectors |
| Biomedical Engineering | SQI (LNR+HNI+Kurt), Spatial Kriging, Cole-Cole Impedance |
| Cybernetics | NMPC (SLSQP, T=10), UKF state estimation |
| Explainable AI | Path Integrated Gradients, Freq-Band Decomposition |
| Digital Twin | Closed-loop simulation, FSCV, LiDAR terrain |
| Safety | Shannon k≤1.75, Fall-reflex (<250μs), Thermal guard |
| Regulatory | FHIR R4 bundles, FDA/EMA model cards, GDPR pseudonymisation |

## WebGL Dashboard Features

The dashboard (`neuroforge/dashboard/index.html`) renders two WebGL canvases via Three.js r158:

1. **Canvas 1 — Biomechanical Subsystem**: Articulated 3D leg (hip→knee→shin→foot) driven by kinematic angles, epidural electrode array, and interactive OrbitControls.
2. **Canvas 2 — Cortical Attractor**: 5×5×5 Hopf vector quiver field, glowing state cursor, orbital trail ribbon, and auto-rotating camera.
3. **Telemetry Strip**:
   - 15-contact spinal epidural array with real-time flexor/extensor contact activation.
   - Live system gauges: SQI, End-to-End Latency, Dopamine concentration, Cole-Cole α biofouling.
   - Explainable AI frequency-band attribution meters (Delta, Theta, Alpha, Mu/Beta, Low-Gamma, High-Gamma).
   - Real-time safety alert ticker with colour-coded status pills.

---

## Bugs Fixed in CANRP-X v3.0

1. **Dashboard State HTTP 404 Resolution**:
   - `SimpleHTTPRequestHandler` isolated requests inside `neuroforge/dashboard/`, which returned 404 when polling `../dashboard_state.json`.
   - Fixed by serving `dashboard_state.json` directly from the dashboard server handler (`_DashboardHandler`) with CORS and cache headers, updating state file output to both paths, and enabling resilient multi-path client fetch.
2. **Kriging Solver Numerical Stability**:
   - `scipy.linalg.solve` failed with LAPACK singular errors on sparse channel configurations.
   - Replaced with regularised least-squares (`np.linalg.lstsq`) and uniform-weight fallback.
3. **Riemannian Procrustes SPD Decomposition**:
   - Replaced `scipy.linalg.sqrtm` and `scipy.linalg.inv` with eigenvalue-decomposition (`_spd_sqrt`) with eigenvalue clamping to prevent complex NaN drift.
4. **Shannon Limit Test Parameters**:
   - Adjusted `test_shannon_safe` test inputs to remain safely beneath the critical `k=1.75` threshold.
5. **MAE Reconstruction Dimensionality**:
   - Fixed `masked_encode` target to match decoded raw patch pixels (`n_channels * patch_samples`) instead of latent embedding features.
6. **XAI Autograd Flow & Tensor Dimensions**:
   - Fixed extra dimension in `model_fn` (`unsqueeze(0).unsqueeze(0)`), removed blocking `torch.no_grad()`, and guarded saliency gradients with `torch.nan_to_num`.
7. **PyTorch UserWarnings Suppression**:
   - Configured `enable_nested_tensor=False` in TransformerEncoders using `norm_first=True`.
   - Detached scalar loss tensors before converting to float.

---

*NeuroForge CANRP-X v3.0 — Research use only. Not approved for clinical deployment.*
