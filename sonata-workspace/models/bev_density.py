"""
Differentiable BEV (bird's-eye-view) density matching loss.

Motivation (from the latent/coordinate adversary dead-ends): the realism gap CD
is blind to is *coverage/density*, exactly what the JSD-BEV metric measures
(LiDiff/LiDPM protocol: discretize the cloud to a BEV histogram, compare
distributions). We target that metric DIRECTLY with a differentiable soft BEV
histogram matched by Jensen-Shannon divergence.

Why this is CD-orthogonal (the whole point): the BEV histogram is invariant to
where a point sits *within* its cell, so the loss gradient only moves points to
fix *cross-cell* density/coverage, not to chase per-point CD correspondences.
It can improve JSD without pulling points off the surface, sidestepping the
adversary-vs-Chamfer conflict.

The histogram uses a separable Gaussian kernel so it is O(N*G), not O(N*G^2),
and fully differentiable w.r.t. point coordinates (soft-splat / KDE style).
"""
from __future__ import annotations

import torch


def soft_bev_hist(points: torch.Tensor, grid: int = 40, bound: float = 1.0,
                  sigma: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable BEV density histogram over the (x, y) plane in [-bound, bound].

    points: (N, 3) or (N, 2) in normalised space.
    Returns: (grid, grid) histogram normalised to sum 1 (a probability map).

    hist[i, j] = sum_n exp(-(x_n - cx_i)^2 / 2σ²) * exp(-(y_n - cy_j)^2 / 2σ²)
    computed as an outer product of the two separable 1-D Gaussian responses.
    """
    xy = points[:, :2]
    centers = torch.linspace(-bound, bound, grid, device=points.device, dtype=points.dtype)
    dx = xy[:, 0:1] - centers[None, :]          # (N, G)
    dy = xy[:, 1:2] - centers[None, :]          # (N, G)
    two_s2 = 2.0 * sigma * sigma
    kx = torch.exp(-(dx * dx) / two_s2)         # (N, G)
    ky = torch.exp(-(dy * dy) / two_s2)         # (N, G)
    hist = kx.t() @ ky                          # (G, G) = sum_n kx[n,i] ky[n,j]
    return hist / (hist.sum() + eps)


def bev_jsd_loss(pred_pts: torch.Tensor, gt_pts: torch.Tensor, grid: int = 40,
                 bound: float = 1.0, sigma: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between the soft BEV histograms of pred and gt."""
    p = soft_bev_hist(pred_pts, grid, bound, sigma, eps).flatten() + eps
    q = soft_bev_hist(gt_pts, grid, bound, sigma, eps).flatten() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return 0.5 * kl_pm + 0.5 * kl_qm
