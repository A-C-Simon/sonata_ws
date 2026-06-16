"""
Localized, permutation-invariant WGAN critic for point-cloud completion.

Motivation (see root-cause analysis): the shipped MultiScalePointDiscriminator
max-pools ~8000 points to a single 640-d vector before scoring. That is the
same single-vector global-pool bottleneck that crippled the project's old VAE
and was only fixed by V3's 32 cross-attention latent tokens. Measured on the
generator, the resulting critic gradient is BOTH ~290x larger than the Chamfer
gradient AND anti-aligned with it (cos(g_cd, g_adv) = -0.28): it pushes points
off-surface to satisfy global scene statistics, regressing CD and even JSD.

This critic mirrors V3's winning inductive bias — it cross-attention-pools the
cloud to K critic tokens, scores each token locally, and mean-aggregates into
the single unbounded scalar WGAN needs. A localized (PatchGAN-analogue) critic
gives a per-region gradient that can be aligned with reducing CD instead of
trading against it. Optionally conditional on a context cloud (the partial
scan), so it judges plausibility *given the scene* rather than generic realism.

LayerNorm only (no BatchNorm) for gradient-penalty / R1 correctness.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention (queries attend to key/value features) + FFN."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + h
        q = q + self.ff(self.norm_ff(q))
        return q


class TokenCritic(nn.Module):
    """
    Cross-attention token critic.

      xyz (B, N, 3)
        → per-point MLP                       (B, N, dim)
        → K learned queries cross-attend      (B, K, dim)   [local patches]
        → per-token scalar                    (B, K)
        → mean over tokens                    (B, 1)        [Wasserstein score]

    If conditional, the K queries also cross-attend to per-point features of a
    context cloud (the partial scan) before scoring, so the critic's notion of
    "real" is conditioned on the input scene.
    """

    def __init__(
        self,
        in_dim: int = 3,
        dim: int = 256,
        num_tokens: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        conditional: bool = False,
    ):
        super().__init__()
        self.conditional = conditional

        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, dim), nn.LayerNorm(dim), nn.LeakyReLU(0.2, inplace=True),
        )
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([_CrossAttnBlock(dim, num_heads) for _ in range(num_layers)])

        if conditional:
            self.ctx_mlp = nn.Sequential(
                nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, dim), nn.LayerNorm(dim), nn.LeakyReLU(0.2, inplace=True),
            )
            self.ctx_blocks = nn.ModuleList([_CrossAttnBlock(dim, num_heads) for _ in range(num_layers)])

        self.token_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        B = xyz.size(0)
        feat = self.point_mlp(xyz)                  # (B, N, dim)
        tokens = self.query.expand(B, -1, -1)       # (B, K, dim)
        for blk in self.blocks:
            tokens = blk(tokens, feat)

        if self.conditional and ctx is not None:
            cfeat = self.ctx_mlp(ctx)
            for blk in self.ctx_blocks:
                tokens = blk(tokens, cfeat)

        per_token = self.token_head(tokens).squeeze(-1)   # (B, K)
        score = per_token.mean(dim=1, keepdim=True)        # (B, 1)
        if return_features:
            return score, [tokens]
        return score


def feature_matching_loss_tokens(real_feats: List[torch.Tensor],
                                 fake_feats: List[torch.Tensor]) -> torch.Tensor:
    """L1 feature matching on per-token critic embeddings (local, not global)."""
    loss = 0.0
    for rf, ff in zip(real_feats, fake_feats):
        loss = loss + torch.nn.functional.l1_loss(ff, rf.detach())
    return loss


def r1_penalty(disc: nn.Module, real: torch.Tensor,
               ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al. 2018): penalise ||∇_x D(x)||^2 on
    *real* samples only. Unlike the WGAN-GP straight-line interpolation, R1
    needs no real↔fake point correspondence, so it is well-defined for
    unordered point sets (the interpolation in the old gradient_penalty mixes
    points by index, which is geometrically meaningless for sets).
    """
    real = real.detach().requires_grad_(True)
    score = disc(real, ctx) if ctx is not None else disc(real)
    grad = torch.autograd.grad(
        outputs=score.sum(), inputs=real, create_graph=True,
    )[0]
    return grad.reshape(real.size(0), -1).pow(2).sum(1).mean()
