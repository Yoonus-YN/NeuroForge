"""
neuroforge/ai/encoder.py
=========================
Self-Supervised Contrastive Encoder for electrophysiological signals.

Two complementary objectives:
  1. Masked Neural Autoencoder (MNA) — reconstruct randomly masked temporal patches
     of multi-channel LFP/ECoG (60% mask ratio, 25 ms patches).
  2. SimCLR Contrastive Learning — InfoNCE loss on augmented temporal views.

Architecture:
  Input  : (N_channels, n_patches, patch_size)
  Encoder: Patch embedding → Transformer → latent z ∈ ℝ^D_MODEL
  Decoder: Linear projection back to patch space (MAE)
  Head   : MLP projection head for contrastive objective

InfoNCE loss:
  L = −log[ exp(sim(zᵢ, zⱼ)/τ) / Σₖ exp(sim(zᵢ, zₖ)/τ) ]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from neuroforge.core.config import (
    N_ECOG_CHANNELS, PATCH_SIZE_MS, MASK_RATIO, FS_ECOG,
    D_MODEL, N_HEADS, N_TRANSFORMER_LAYERS, FFN_DIM, DROPOUT,
    SIMCLR_TEMP,
)
from neuroforge.core.logger import get_logger

log = get_logger(__name__)

PATCH_SAMPLES = int(PATCH_SIZE_MS * FS_ECOG / 1000)   # samples per patch


# ---------------------------------------------------------------------------
# Temporal augmentations for SimCLR views
# ---------------------------------------------------------------------------
class TemporalAugment(nn.Module):
    """Random augmentations: time-shift jitter, channel dropout, band-stop."""

    def __init__(self, jitter_frac: float = 0.05, ch_drop_prob: float = 0.2):
        super().__init__()
        self.jitter_frac = jitter_frac
        self.ch_drop_prob = ch_drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time)"""
        # Channel dropout
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device)
                > self.ch_drop_prob).float()
        x = x * mask
        # Additive Gaussian noise
        x = x + 0.05 * torch.randn_like(x)
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split time-series into non-overlapping 25 ms patches and linearly embed."""

    def __init__(self, n_channels: int, patch_samples: int, d_model: int):
        super().__init__()
        self.patch_samples = patch_samples
        self.proj = nn.Linear(n_channels * patch_samples, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, n_samples)
        Returns: (batch, n_patches, d_model)
        """
        b, c, t = x.shape
        n_patches = t // self.patch_samples
        # Truncate to complete patches
        x = x[:, :, : n_patches * self.patch_samples]
        # Reshape: (batch, n_patches, channels * patch_samples)
        x = x.reshape(b, n_patches, c * self.patch_samples)
        return self.proj(x)    # (batch, n_patches, d_model)


# ---------------------------------------------------------------------------
# Main Encoder
# ---------------------------------------------------------------------------
class NeuroEncoder(nn.Module):
    """
    Asymmetric Transformer encoder for masked autoencoding + contrastive learning.
    """

    def __init__(
        self,
        n_channels: int = N_ECOG_CHANNELS,
        patch_samples: int = PATCH_SAMPLES,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_TRANSFORMER_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_channels, patch_samples, d_model)
        self.patch_samples = patch_samples
        self.d_model = d_model

        # Learnable positional encodings
        self.pos_emb = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        # MAE decoder (lightweight)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, n_channels * patch_samples),
        )

        # SimCLR projection head
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, proj_dim),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encoding pass.

        Args:
            x: (batch, n_channels, n_samples)

        Returns:
            (tokens, z_mean) — patch tokens (b, n_patches, d_model),
                               mean-pooled latent (b, d_model)
        """
        tokens = self.patch_embed(x)                      # (b, n_patches, d_model)
        n_patches = tokens.size(1)
        pos = torch.arange(n_patches, device=x.device)
        tokens = tokens + self.pos_emb(pos)
        out = self.transformer(tokens)
        z = out.mean(dim=1)                               # (b, d_model)
        return out, z

    def masked_encode(
        self,
        x: torch.Tensor,
        mask_ratio: float = MASK_RATIO,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked autoencoder forward pass (60% of patches masked).

        Returns:
            (tokens_visible, mask_bool, target_patches)
        """
        tokens = self.patch_embed(x)
        b, n_patches, d = tokens.shape
        pos = torch.arange(n_patches, device=x.device)
        tokens = tokens + self.pos_emb(pos)

        # Random mask
        n_mask = int(mask_ratio * n_patches)
        noise = torch.rand(b, n_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask_ids = ids_shuffle[:, :n_mask]
        keep_ids = ids_shuffle[:, n_mask:]

        # Build full sequence: keep visible, replace masked with mask_token
        tokens_full = tokens.clone()
        tokens_full.scatter_(
            1,
            mask_ids.unsqueeze(-1).expand(-1, -1, d),
            self.mask_token.expand(b, n_mask, d),
        )

        out = self.transformer(tokens_full)     # all patches (with mask tokens)
        mask_bool = torch.zeros(b, n_patches, dtype=torch.bool, device=x.device)
        mask_bool.scatter_(1, mask_ids, True)

        # Target: raw patch values (before linear projection), shape (b, n_patches, ch*ps)
        b_sz, c_sz, t_sz = x.shape
        n_p = t_sz // self.patch_samples
        x_trunc = x[:, :, :n_p * self.patch_samples]
        target = x_trunc.reshape(b_sz, n_p, c_sz * self.patch_samples)  # (b, n_patches, ch*ps)
        return out, mask_bool, target

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "encode",
        mask_ratio: float = MASK_RATIO,
    ) -> dict[str, torch.Tensor]:
        """
        mode="encode" → returns latent z and projection h.
        mode="mae"    → returns reconstruction loss.
        mode="simclr" → returns projection h (for InfoNCE).
        """
        if mode == "mae":
            out, mask_bool, target = self.masked_encode(x, mask_ratio)
            recon = self.decoder(out)                # (b, n_patches, ch*ps)
            loss = F.mse_loss(recon[mask_bool], target[mask_bool].detach())
            return {"mae_loss": loss, "recon": recon}

        _, z = self.encode(x)

        if mode == "simclr":
            h = F.normalize(self.proj_head(z), dim=-1)
            return {"z": z, "h": h}

        return {"z": z}


# ---------------------------------------------------------------------------
# InfoNCE / SimCLR Loss
# ---------------------------------------------------------------------------
def info_nce_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    temperature: float = SIMCLR_TEMP,
) -> torch.Tensor:
    """
    Compute InfoNCE (NT-Xent) loss between two views.

    Args:
        h1, h2: (batch, proj_dim) — L2-normalised projections of two augmented views.
        temperature: τ parameter.

    Returns:
        Scalar InfoNCE loss.
    """
    batch_size = h1.size(0)
    z = torch.cat([h1, h2], dim=0)   # (2B, proj_dim)
    sim = torch.mm(z, z.t()) / temperature   # (2B, 2B)

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=h1.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i + B) and (i + B, i)
    labels = torch.arange(batch_size, device=h1.device)
    labels = torch.cat([labels + batch_size, labels])

    loss = F.cross_entropy(sim, labels)
    return loss
