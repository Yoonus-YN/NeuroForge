"""
neuroforge/ai/decoder.py
=========================
Temporal Patch-Transformer + Mamba-style SSM ensemble decoder for continuous
kinematic trajectory estimation from cortical latent embeddings.

Architecture:
  Stream A — Temporal Patch-Transformer: multi-head self-attention over
             sequential latent embeddings with LiDAR terrain context fusion.
  Stream B — Bi-directional Linear State-Space Model (Mamba-inspired):
             O(N) recurrent trajectory tracking.
  Fusion  — Dynamic Bayesian Model Averaging weighted by inverse predictive
             entropy of each stream.

Output: joint angle trajectory (hip, knee, ankle) in radians.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from neuroforge.core.config import (
    D_MODEL, N_HEADS, N_TRANSFORMER_LAYERS, FFN_DIM, DROPOUT,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)

N_JOINTS = 3      # hip, knee, ankle


# ---------------------------------------------------------------------------
# Stream A: Temporal Patch-Transformer
# ---------------------------------------------------------------------------
class TerrainFusion(nn.Module):
    """
    Fuse LiDAR distance readings (8 sectors) into a terrain context vector,
    then add to latent token sequence via cross-attention.
    """

    def __init__(self, lidar_dim: int = 8, d_model: int = D_MODEL):
        super().__init__()
        self.terrain_enc = nn.Sequential(
            nn.Linear(lidar_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """
        z     : (batch, seq_len, d_model)
        lidar : (batch, 8) — LiDAR distance readings in metres.
        Returns: (batch, seq_len, d_model) with terrain context added.
        """
        ctx = self.terrain_enc(lidar)          # (batch, d_model)
        ctx = ctx.unsqueeze(1)                 # (batch, 1, d_model)
        return z + ctx                         # broadcast addition


class TemporalPatchTransformer(nn.Module):
    """Self-attention over sequential latent embeddings + terrain context."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_TRANSFORMER_LAYERS, ffn_dim: int = FFN_DIM,
                 dropout: float = DROPOUT, n_joints: int = N_JOINTS):
        super().__init__()
        self.terrain_fusion = TerrainFusion(d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_joints),
        )

    def forward(
        self,
        z_seq: torch.Tensor,       # (batch, seq_len, d_model)
        lidar: torch.Tensor,       # (batch, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (angles, entropy)."""
        z = self.terrain_fusion(z_seq, lidar)
        out = self.transformer(z)
        pooled = out.mean(dim=1)                          # (batch, d_model)
        angles = self.regressor(pooled)                   # (batch, N_JOINTS)

        # Predictive entropy (approximated via MC mean over batch)
        log_var = torch.log(torch.var(angles, dim=0, correction=0) + 1e-8)
        entropy = float(torch.mean(log_var).item())

        return angles, torch.tensor(entropy)


# ---------------------------------------------------------------------------
# Stream B: Linear State-Space Model (Mamba-inspired, bi-directional)
# ---------------------------------------------------------------------------
class BidirectionalSSM(nn.Module):
    """
    Simplified Mamba-inspired bi-directional state-space decoder.
    Uses learned A, B, C matrices for O(N) trajectory tracking.
    """

    def __init__(self, d_model: int = D_MODEL, d_state: int = 16,
                 n_joints: int = N_JOINTS):
        super().__init__()
        self.d_state = d_state

        # Forward pass matrices
        self.A_fwd = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B_fwd = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C_fwd = nn.Parameter(torch.randn(n_joints, d_state) * 0.01)

        # Backward pass matrices
        self.A_bwd = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B_bwd = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C_bwd = nn.Parameter(torch.randn(n_joints, d_state) * 0.01)

        self.output_proj = nn.Linear(n_joints * 2, n_joints)

    def _scan(
        self,
        z_seq: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Linear recurrence scan: h_t = A h_{t-1} + B z_t."""
        b, t, _ = z_seq.shape
        h = torch.zeros(b, self.d_state, device=z_seq.device)
        outputs = []
        for i in range(t):
            h = torch.tanh(h @ A.T + z_seq[:, i, :] @ B.T)
            y = h @ C.T
            outputs.append(y)
        return torch.stack(outputs, dim=1)   # (b, t, n_joints)

    def forward(
        self,
        z_seq: torch.Tensor,       # (batch, seq_len, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (angles, entropy)."""
        fwd = self._scan(z_seq, self.A_fwd, self.B_fwd, self.C_fwd)
        bwd = self._scan(torch.flip(z_seq, [1]), self.A_bwd, self.B_bwd, self.C_bwd)
        bwd = torch.flip(bwd, [1])

        combined = torch.cat([fwd, bwd], dim=-1)  # (b, t, 2*n_joints)
        angles = self.output_proj(combined).mean(dim=1)   # (b, n_joints)

        log_var = torch.log(torch.var(angles, dim=0, correction=0) + 1e-8)
        entropy = float(torch.mean(log_var).item())

        return angles, torch.tensor(entropy)


# ---------------------------------------------------------------------------
# Ensemble: Dynamic Bayesian Model Averaging
# ---------------------------------------------------------------------------
class EnsembleDecoder(nn.Module):
    """
    Weighted ensemble of Transformer + SSM via inverse-entropy BMA.
    """

    def __init__(self):
        super().__init__()
        self.transformer = TemporalPatchTransformer()
        self.ssm = BidirectionalSSM()

    def forward(
        self,
        z_seq: torch.Tensor,    # (batch, seq_len, d_model)
        lidar: torch.Tensor,    # (batch, 8)
    ) -> dict[str, torch.Tensor]:
        angles_t, ent_t = self.transformer(z_seq, lidar)
        angles_s, ent_s = self.ssm(z_seq)

        # Inverse entropy weighting (lower uncertainty → higher weight)
        w_t = 1.0 / (abs(ent_t.item()) + 1e-6)
        w_s = 1.0 / (abs(ent_s.item()) + 1e-6)
        total = w_t + w_s
        w_t /= total
        w_s /= total

        angles_fused = w_t * angles_t + w_s * angles_s

        return {
            "angles_deg": torch.rad2deg(angles_fused),
            "angles_rad": angles_fused,
            "weight_transformer": w_t,
            "weight_ssm": w_s,
            "entropy_transformer": float(ent_t),
            "entropy_ssm": float(ent_s),
        }


# ---------------------------------------------------------------------------
# Continual Learning: Elastic Weight Consolidation (EWC)
# ---------------------------------------------------------------------------
class EWCRegularizer:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Penalty:  L_EWC = (λ/2) Σᵢ Fᵢ (θᵢ − θᵢ*)²

    where Fᵢ is the Fisher information for parameter i,
    θᵢ* is the parameter value after the previous session.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher: dict[str, torch.Tensor] = {}
        self.params_star: dict[str, torch.Tensor] = {}

    def register_task(
        self,
        data_loader: list[torch.Tensor],
        loss_fn,
        n_batches: int = 10,
    ) -> None:
        """Compute Fisher information matrix from current task data."""
        self.params_star = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        for i, batch in enumerate(data_loader[:n_batches]):
            self.model.zero_grad()
            output = loss_fn(batch)
            output.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2

        for n in self.fisher:
            self.fisher[n] /= max(min(n_batches, len(data_loader)), 1)

        log.info("EWC Fisher information registered for %d parameters.", len(self.fisher))

    def penalty(self) -> torch.Tensor:
        """Compute EWC regularisation penalty."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params_star[n]) ** 2).sum()
        return (self.ewc_lambda / 2.0) * loss
