"""
neuroforge/core/config.py
==========================
Global configuration dataclasses for all CANRP-X v3.0 subsystems.
All numeric constants are grounded in clinical and embedded-silicon specifications
from the system architecture document.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Sampling & timing
# ---------------------------------------------------------------------------
FS_ECOG: int = 2000          # μECoG / LFP sampling rate  (Hz)
FS_EMG: int = 2000           # sEMG sampling rate          (Hz)
FS_IMU: int = 200            # IMU sampling rate            (Hz)
FS_FSCV: int = 10            # FSCV voltammetry rate        (Hz)

DT_FAST: float = 250e-6      # Ultra-fast loop period       (s) — 4 kHz
DT_CLOSED: float = 10e-3     # Fast closed-loop period      (s) — 100 Hz
DT_INTER: float = 100e-3     # Intermediate loop period     (s) — 10 Hz

N_ECOG_CHANNELS: int = 16    # Subdural μECoG channels
N_EMG_CHANNELS: int = 8      # sEMG intramuscular channels
N_SPINAL_CONTACTS: int = 15  # Epidural array contacts

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
SEED: int = 42               # Deterministic random seed for reproducibility

# ---------------------------------------------------------------------------
# SQI thresholds
# ---------------------------------------------------------------------------
SQI_THRESHOLD: float = 0.75
SQI_W1: float = 0.45         # LNR weight
SQI_W2: float = 0.35         # HNI weight
SQI_W3: float = 0.20         # Kurtosis weight

# ---------------------------------------------------------------------------
# Shannon charge-density safety
# ---------------------------------------------------------------------------
SHANNON_K_LIMIT: float = 1.75     # k = log(D) + log(A) <= 1.75
SAFE_CURRENT_AMPLITUDE_UA: float = 100.0   # μA max safe per contact

# ---------------------------------------------------------------------------
# Cole-Cole tissue impedance
# ---------------------------------------------------------------------------
COLE_R_INF: float = 200.0         # Ω — high-freq resistance
COLE_R0: float = 800.0            # Ω — DC resistance
COLE_TAU: float = 1e-5            # s  — time constant
COLE_ALPHA_MIN: float = 0.70      # Dispersion factor floor (biofouling alert)

# ---------------------------------------------------------------------------
# FSCV dopamine / serotonin
# ---------------------------------------------------------------------------
FSCV_SWEEP_RATE: float = 400.0    # V/s
FSCV_V_MIN: float = -0.4          # V
FSCV_V_MAX: float = 1.3           # V
DOPAMINE_BASELINE_NM: float = 80.0
SEROTONIN_BASELINE_NM: float = 40.0

# ---------------------------------------------------------------------------
# IMU / LiDAR
# ---------------------------------------------------------------------------
IMU_TRIP_THRESHOLD_DPS: float = 300.0   # °/s — fall-reflex trigger
LIDAR_STAIR_HEIGHT_CM: float = 15.0     # cm  — proactive terrain switch
BRACE_TORQUE_NM: float = 85.0           # Nm  — co-contraction torque

# ---------------------------------------------------------------------------
# Izhikevich CPG
# ---------------------------------------------------------------------------
IZH_A: float = 0.02
IZH_B: float = 0.20
IZH_C: float = -50.0
IZH_D: float = 2.0
IZH_V_THRESH: float = 30.0          # mV — spike threshold
IZH_V_RESET: float = -65.0          # mV — membrane rest
CPG_N_FLEXOR: int = 20
CPG_N_EXTENSOR: int = 20
CPG_MUTUAL_INHIBITION: float = -1.5  # mA — cross-inhibitory synaptic weight

# ---------------------------------------------------------------------------
# Triplet STDP
# ---------------------------------------------------------------------------
STDP_A2_PLUS: float = 5e-3
STDP_A3_PLUS: float = 6.2e-3
STDP_TAU_PLUS: float = 16.8e-3   # s
STDP_TAU_Y: float = 114e-3       # s
STDP_A2_MINUS: float = 7e-3
STDP_TAU_MINUS: float = 33.7e-3  # s

# ---------------------------------------------------------------------------
# Transformer / SSM decoder
# ---------------------------------------------------------------------------
PATCH_SIZE_MS: int = 25         # ms — temporal patch width
MASK_RATIO: float = 0.60        # MAE masking ratio
D_MODEL: int = 128              # Transformer embedding dimension
N_HEADS: int = 4
N_TRANSFORMER_LAYERS: int = 3
FFN_DIM: int = 256
DROPOUT: float = 0.1
SIMCLR_TEMP: float = 0.07       # InfoNCE temperature τ

# ---------------------------------------------------------------------------
# NMPC
# ---------------------------------------------------------------------------
NMPC_HORIZON: int = 10          # prediction steps
NMPC_DT: float = 0.01           # s
NMPC_KP: float = 120.0          # proportional stiffness  Nm/rad
NMPC_KD: float = 12.0           # derivative damping       Nm·s/rad
NMPC_TAU_MIN: float = -150.0    # Nm
NMPC_TAU_MAX: float = 150.0     # Nm

# ---------------------------------------------------------------------------
# EWC continual learning
# ---------------------------------------------------------------------------
EWC_LAMBDA: float = 400.0       # Fisher regularization strength

# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------
THERMAL_DELTA_T_MAX: float = 0.8    # °C — throttle threshold (ISO 14708-1)

# ---------------------------------------------------------------------------
# FHIR / telemetry
# ---------------------------------------------------------------------------
FHIR_PATIENT_ID: str = "NEUROFORGE-SCI-001"
FHIR_DEVICE_ID: str = "CANRP-X-v3.0"
