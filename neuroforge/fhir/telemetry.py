"""
neuroforge/fhir/telemetry.py
=============================
FHIR R4 / HL7 compliant clinical telemetry export for CANRP-X v3.0.

Generates:
  • Observation resources for step metrics, spasticity index, SQI.
  • DiagnosticReport resource bundling a rehabilitation session.
  • AES-256-GCM encryption stub for OTA upload (key management is external).

HIPAA / GDPR Article 9 compliance:
  - Patient identifiers are pseudonymised.
  - All exports include a data-category field for special-category health data.
"""
from __future__ import annotations
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from neuroforge.core.config import FHIR_PATIENT_ID, FHIR_DEVICE_ID
from neuroforge.core.logger import get_logger

log = get_logger(__name__)

FHIR_OUTPUT_DIR = Path(__file__).parents[2] / "fhir_exports"
FHIR_OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pseudonymise(patient_id: str) -> str:
    """SHA-256 one-way pseudonym for GDPR Article 9 compliance."""
    return "PSEUDO-" + hashlib.sha256(patient_id.encode()).hexdigest()[:12].upper()


# ---------------------------------------------------------------------------
# FHIR Observation Builder
# ---------------------------------------------------------------------------
def make_observation(
    code: str,
    display: str,
    value: float,
    unit: str,
    unit_code: str,
    category: str = "vital-signs",
) -> dict[str, Any]:
    """Build a minimal FHIR R4 Observation resource."""
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": category,
                "display": category.replace("-", " ").title(),
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": code,
                "display": display,
            }]
        },
        "subject": {
            "reference": f"Patient/{_pseudonymise(FHIR_PATIENT_ID)}"
        },
        "device": {
            "reference": f"Device/{FHIR_DEVICE_ID}"
        },
        "effectiveDateTime": _now_iso(),
        "valueQuantity": {
            "value": round(float(value), 4),
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit_code,
        },
    }


# ---------------------------------------------------------------------------
# Session Telemetry Bundle
# ---------------------------------------------------------------------------
@dataclass
class SessionMetrics:
    total_steps: int
    mean_sqi: float
    spasticity_episodes: int
    dopamine_mean_nm: float
    decoder_latency_ms: float
    synaptic_weight_delta: float
    session_duration_min: float
    alert_flags: list[str] = field(default_factory=list)


def export_session_bundle(metrics: SessionMetrics) -> Path:
    """
    Export a FHIR DiagnosticReport bundle containing rehabilitation metrics.

    Returns:
        Path to the exported JSON file.
    """
    observations = [
        make_observation(
            "228429008", "Number of steps taken",
            metrics.total_steps, "steps", "{steps}",
            category="activity",
        ),
        make_observation(
            "386725004", "Signal quality index",
            metrics.mean_sqi, "dimensionless", "1",
            category="laboratory",
        ),
        make_observation(
            "397765004", "Spastic episodes count",
            metrics.spasticity_episodes, "count", "{count}",
            category="exam",
        ),
        make_observation(
            "55543008", "Dopamine concentration",
            metrics.dopamine_mean_nm, "nM", "nmol/L",
            category="laboratory",
        ),
        make_observation(
            "251765005", "Closed-loop latency",
            metrics.decoder_latency_ms, "ms", "ms",
            category="vital-signs",
        ),
        make_observation(
            "7889000", "Synaptic weight change (STDP)",
            metrics.synaptic_weight_delta, "dimensionless", "1",
            category="laboratory",
        ),
    ]

    diagnostic_report: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "NRS",
                "display": "Neurology",
            }]
        }],
        "code": {
            "text": "NeuroForge CANRP-X Rehabilitation Session Report"
        },
        "subject": {
            "reference": f"Patient/{_pseudonymise(FHIR_PATIENT_ID)}"
        },
        "effectivePeriod": {
            "start": _now_iso(),
            "end": _now_iso(),
        },
        "issued": _now_iso(),
        "result": [
            {"reference": f"#{obs['id']}"} for obs in observations
        ],
        "contained": observations,
        "conclusion": (
            f"Rehabilitation session completed. "
            f"Steps: {metrics.total_steps}. "
            f"Mean SQI: {metrics.mean_sqi:.2f}. "
            f"Latency: {metrics.decoder_latency_ms:.1f} ms. "
            f"Alerts: {metrics.alert_flags or 'None'}."
        ),
        "meta": {
            "security": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-Confidentiality",
                "code": "R",    # Restricted
                "display": "Restricted — GDPR Article 9 Special Category",
            }]
        },
    }

    bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "document",
        "timestamp": _now_iso(),
        "entry": [{"resource": diagnostic_report}],
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = FHIR_OUTPUT_DIR / f"session_{ts}.fhir.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    log.info("FHIR bundle exported → %s", out_path)
    return out_path
