"""
Latent-space adversary for the VAE-GAN (the structural cure for the
coordinate-space anti-alignment).

Measured problem: any critic scoring realism of decoded XYZ gives a generator
gradient anti-aligned with Chamfer (cos -0.45..-0.70 for both conditional and
unconditional token critics, Stage-1/Stage-2) -- "look more real" and "land on
the GT surface" fight over the same coordinates.

Fix (LION / latent-GAN style): put the adversary on the VAE's latent TOKENS,
with the encoder AND decoder frozen. The critic scores token realism and a
small bounded latent refiner moves tokens within the decoder's learned manifold.
The decoder maps any latent to an on-manifold (plausible) cloud, so the
adversarial gradient -- routed through the frozen decoder + low-dim latent --
can only produce realistic scene-level variation rather than off-surface
per-point shifts. CD is protected by the frozen decoder + a latent-offset leash.

All LayerNorm (no BatchNorm) for R1 correctness.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention over the L latent tokens + FFN."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class LatentTokenCritic(nn.Module):
    """
    WGAN critic on latent tokens (B, L, token_dim):
      embed -> self-attention over the L tokens -> per-token score -> mean -> (B,1).
    Permutation-equivariant over tokens; mean aggregation gives the single
    unbounded Wasserstein scalar.
    """

    def __init__(self, token_dim: int, dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, num_tokens: int = 32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(token_dim, dim), nn.LayerNorm(dim), nn.LeakyReLU(0.2, inplace=True),
        )
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([_SelfAttnBlock(dim, num_heads) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos[:, : tokens.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1).mean(dim=1, keepdim=True)  # (B, 1)


class LatentRefiner(nn.Module):
    """Bounded residual on latent tokens: refined = tokens + max_dz*tanh(head(.)).
    Near-identity at init (zero-init head)."""

    def __init__(self, token_dim: int, dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, num_tokens: int = 32, max_dz: float = 0.5):
        super().__init__()
        self.max_dz = max_dz
        self.embed = nn.Sequential(
            nn.Linear(token_dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([_SelfAttnBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, token_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens) + self.pos[:, : tokens.size(1)]
        for blk in self.blocks:
            x = blk(x)
        dz = self.max_dz * torch.tanh(self.head(self.norm(x)))
        return tokens + dz, dz


def r1_penalty_latent(critic: nn.Module, real_tokens: torch.Tensor) -> torch.Tensor:
    """R1 penalty on real latent tokens (no real/fake correspondence needed)."""
    real = real_tokens.detach().requires_grad_(True)
    score = critic(real)
    grad = torch.autograd.grad(outputs=score.sum(), inputs=real, create_graph=True)[0]
    return grad.reshape(real.size(0), -1).pow(2).sum(1).mean()
