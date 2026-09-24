"""
neuroforge/ai/domain_adapt.py
==============================
Domain adaptation pipeline for cross-subject and cross-session transfer.

Methods:
  1. Riemannian Procrustes Alignment — aligns daily covariance matrices on the
     SPD manifold using affine-invariant distance.
  2. Maximum Mean Discrepancy (MMD) — kernel-based distribution matching loss
     for adversarial domain alignment.

Both are applied without retraining deep weights, enabling zero-shot
deployment to new clinical subjects.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from neuroforge.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Riemannian Procrustes Alignment
# ---------------------------------------------------------------------------
def _spd_sqrt(M: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Compute symmetric positive-definite matrix (square root or inverse square root)
    via eigendecomposition. Robust to near-singular matrices via eigenvalue clipping.
    """
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-8, None)   # clip negative/zero eigenvalues
    if inverse:
        factor = 1.0 / np.sqrt(vals)
    else:
        factor = np.sqrt(vals)
    return (vecs * factor) @ vecs.T


def riemannian_procrustes_align(
    cov_source: np.ndarray,   # (n_ch, n_ch) — source session covariance (SPD)
    cov_target: np.ndarray,   # (n_ch, n_ch) — target session covariance (SPD)
    X_source: np.ndarray,     # (n_samples, n_ch) — source-session data
) -> np.ndarray:
    """
    Affine-invariant Riemannian Procrustes alignment.

    Transforms X_source so its covariance matches cov_target:
        X_aligned = X_source @ W
        W = C_s^{-1/2} @ C_t^{1/2}

    Args:
        cov_source: (n_ch, n_ch) source covariance matrix.
        cov_target: (n_ch, n_ch) target covariance matrix.
        X_source  : (n_samples, n_ch) source data.

    Returns:
        X_aligned: (n_samples, n_ch) aligned data.
    """
    eps = 1e-6 * np.eye(cov_source.shape[0])
    Cs = cov_source + eps
    Ct = cov_target + eps

    Cs_sqrt_inv = _spd_sqrt(Cs, inverse=True)
    Ct_sqrt = _spd_sqrt(Ct, inverse=False)

    W = Cs_sqrt_inv @ Ct_sqrt     # (n_ch, n_ch)
    X_aligned = X_source @ W

    log.debug(
        "Riemannian Procrustes alignment complete. "
        "||W − I||_F = %.4f",
        float(np.linalg.norm(W - np.eye(W.shape[0]), "fro")),
    )
    return X_aligned


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) Loss
# ---------------------------------------------------------------------------
def _gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """k(x, y) = exp(−||x − y||² / (2σ²))"""
    diff = x.unsqueeze(1) - y.unsqueeze(0)       # (m, n, d)
    sq_dist = (diff ** 2).sum(-1)                 # (m, n)
    return torch.exp(-sq_dist / (2 * sigma ** 2))


def mmd_loss(
    z_source: torch.Tensor,    # (m, d) source domain latents
    z_target: torch.Tensor,    # (n, d) target domain latents
    sigmas: list[float] | None = None,
) -> torch.Tensor:
    """
    Unbiased multi-kernel Maximum Mean Discrepancy estimator.

    L_MMD = E[k(zₛ,zₛ)] − 2E[k(zₛ,zₜ)] + E[k(zₜ,zₜ)]

    Args:
        z_source: Source-domain latent embeddings.
        z_target: Target-domain latent embeddings.
        sigmas  : List of kernel bandwidths for multi-scale MMD.

    Returns:
        Scalar MMD² loss.
    """
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0, 5.0]

    mmd = torch.tensor(0.0, device=z_source.device)
    for s in sigmas:
        K_ss = _gaussian_kernel(z_source, z_source, sigma=s)
        K_tt = _gaussian_kernel(z_target, z_target, sigma=s)
        K_st = _gaussian_kernel(z_source, z_target, sigma=s)
        mmd += K_ss.mean() - 2.0 * K_st.mean() + K_tt.mean()

    return mmd / len(sigmas)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (DANN)
# ---------------------------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class DomainClassifier(nn.Module):
    """
    Domain discriminator for DANN adversarial alignment.
    Input: latent z ∈ ℝ^D_MODEL → binary domain label (0=source, 1=target).
    """

    def __init__(self, d_model: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply gradient reversal then classify domain.
        z: (batch, d_model)
        Returns: (batch, 1) logits.
        """
        z_rev = GradientReversal.apply(z, alpha)
        return self.net(z_rev)
