"""
CD-leashed residual refiner (Stage 2) for the VAE-GAN.

Rationale (from the eval): the aligned token critic produces a *healthy*
perception-distortion tradeoff -- it improves JSD/coverage but moves points off
the surface, regressing CD/Hausdorff/F/IoU. Stage 2 keeps the CD-optimal Stage-1
decoder frozen and learns only a small, *bounded* per-point offset driven by the
aligned critic. Because the offset is bounded (tanh * max_offset) and penalised
(||offset||^2), CD cannot regress beyond a leash, while the critic is free to add
the high-frequency / density structure that lowers JSD.

The refiner gives each decoded point access to the frozen latent tokens via
cross-attention (the same inductive bias as the decoder), so it can place detail
using regional scene context rather than per-point-in-isolation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.point_cloud_vae import DecoderBlock, PointCloudVAE


class ResidualRefiner(nn.Module):
    """
    base points (B, N, 3) + latent tokens (B, L, D) -> bounded offsets (B, N, 3).

    refined = base + max_offset * tanh(head(cross_attn(embed(base), tokens)))
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, num_blocks: int = 2,
                 max_offset: float = 0.1):
        super().__init__()
        self.max_offset = max_offset
        self.embed = nn.Sequential(
            nn.Linear(3, dim), nn.LayerNorm(dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([DecoderBlock(dim, num_heads) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3)
        # start as near-identity: tiny initial offsets
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """base: (B,N,3) or (N,3); tokens: (B,L,D) or (L,D). Returns same shape as base."""
        single = base.dim() == 2
        if single:
            base = base.unsqueeze(0)
            tokens = tokens.unsqueeze(0)
        q = self.embed(base)
        for blk in self.blocks:
            q = blk(q, tokens)
        offset = self.max_offset * torch.tanh(self.head(self.norm(q)))
        refined = base + offset
        if single:
            refined = refined.squeeze(0)
            offset = offset.squeeze(0)
        return refined, offset


class RefinedVAE(torch.nn.Module):
    """
    Eval-time wrapper presenting the PointCloudVAE interface (encode / decode /
    reparameterize / forward) for a frozen Stage-1 VAE + Stage-2 ResidualRefiner,
    so the existing evaluation/diff machinery works unchanged.
    """

    def __init__(self, base_vae, refiner: ResidualRefiner):
        super().__init__()
        self.base = base_vae
        self.refiner = refiner
        self.num_latent_tokens = base_vae.num_latent_tokens
        self.token_dim = base_vae.token_dim
        self.latent_dim = base_vae.latent_dim

    def encode(self, points):
        return self.base.encode(points)

    @staticmethod
    def reparameterize(mu, logvar):
        return PointCloudVAE.reparameterize(mu, logvar)

    def decode(self, z):
        base = self.base.decode(z)
        single = z.dim() == 1
        zt = z.view(self.num_latent_tokens, self.token_dim) if single \
            else z.view(z.size(0), self.num_latent_tokens, self.token_dim)
        tokens = self.base.token_up(zt)
        refined, _ = self.refiner(base, tokens)
        return refined

    def forward(self, points):
        mu, logvar = self.encode(points)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
