"""
Diffusion Module for Semantic Scene Completion

Implements the diffusion process for point cloud completion,
following LiDiff's point-wise local approach with Sonata encoder conditioning.
"""

import math
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Default shell size / scene bound (matches typical 20 m local KITTI crop in CLI scripts).
DEFAULT_TRAIN_MAX_POINTS = 8000
QUERY_MAX_RADIUS = 20.0
# Horizontal distance from sensor below which we do NOT place synthetic queries (blind / unstable NN-GT).
QUERY_MIN_RADIUS = 3.0
# Stabilize diffusion: cap predicted ε to avoid rare MSE / pred_x0 blow-ups (training + sampling).
PRED_NOISE_CLAMP = 10.0
# Skip t∈[0,9): very low noise makes sqrt(ᾱ) tiny and x0-from-ε numerically unstable.
TRAIN_TIMESTEP_MIN = 10


def build_query_shell(
    partial_coords: torch.Tensor,
    num_query: int = 8000,
    radius: float = 20.0,
    min_radius: float = QUERY_MIN_RADIUS,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Concatenate observed partial voxels with extra queries in a vertical cylinder.

    Design (minimal change vs full redesign): keep **cylinder + uniform-style sampling** so
    training and inference stay simple; avoid surface-aware query placement (harder, needs
    extra modules). Extra queries use **r_xy in [min_radius, radius]** so nothing is sampled
    inside the typical sensor blind / ego cylinder where LiDAR has no returns and NN GT is
    inconsistent (artifact blob near origin).

    We do **not** normalize coordinates here: Sonata and KNN conditioning expect metric
    ``partial_scan['coord']``; normalizing only the diffusion branch would desync encodings
    unless the encoder and all KNN trees were redesigned.

    Coordinate diffusion scale (meters vs N(0,1) noise) is imperfect; fixing it would mean
    global norm/denorm — skipped for safety; see module docstring / training notes.
    """
    if partial_coords.numel() == 0:
        raise ValueError("build_query_shell: partial_coords is empty")
    device = device or partial_coords.device
    partial_coords = partial_coords.to(device=device, dtype=torch.float32)
    R = float(radius)
    r0 = float(min_radius)
    if r0 >= R:
        raise ValueError(f"build_query_shell: min_radius ({r0}) must be < radius ({R})")
    u = torch.rand(num_query, 2, device=device, generator=generator, dtype=torch.float32)
    # Spec: r = sqrt(u) * (R - r_min) + r_min  →  annulus excluding [0, r_min).
    r_xy = torch.sqrt(u[:, 0].clamp(min=1e-8)) * (R - r0) + r0
    theta = 2.0 * math.pi * u[:, 1]
    xy = torch.stack([r_xy * torch.cos(theta), r_xy * torch.sin(theta)], dim=-1)
    z = (torch.rand(num_query, device=device, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * R
    extra = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    return torch.cat([partial_coords, extra], dim=0)


def build_shell_coords_single(
    partial_coords: torch.Tensor,
    num_query_extra: int = DEFAULT_TRAIN_MAX_POINTS,
    target_shell_total: Optional[int] = None,
    max_radius: float = QUERY_MAX_RADIUS,
    min_radius: float = QUERY_MIN_RADIUS,
    query_voxel_size: float = 0.15,
    rng: Optional[int] = None,
) -> torch.Tensor:
    """
    CPU-side shell for inference scripts. ``query_voxel_size`` is kept only for
    backward-compatible CLI; queries are sampled continuously like training.
    ``min_radius`` must match training (no queries inside sensor blind zone).
    """
    _ = query_voxel_size  # legacy API
    t = partial_coords.detach().float().cpu()
    g = torch.Generator(device="cpu")
    if rng is not None:
        g.manual_seed(int(rng))
    n_q = int(num_query_extra)
    if target_shell_total is not None:
        n_q = max(0, int(target_shell_total) - int(t.shape[0]))
    return build_query_shell(
        t,
        num_query=n_q,
        radius=float(max_radius),
        min_radius=float(min_radius),
        device=torch.device("cpu"),
        generator=g,
    )


def farthest_point_sample(coords: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    FPS indices (N,) -> (npoint,) for coords (N, 3). Replaces uniform linewise
    downsampling so coarser scales still cover the frustum / scene geometry.
    """
    N = coords.shape[0]
    if N <= npoint:
        return torch.arange(N, device=coords.device, dtype=torch.long)
    device = coords.device
    idx = torch.zeros(npoint, dtype=torch.long, device=device)
    dist = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device) % N
    farthest = int(farthest.item())
    c = coords.float()
    for i in range(npoint):
        idx[i] = farthest
        cent = c[farthest : farthest + 1]
        d = torch.sum((c - cent) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        farthest = int(torch.argmax(dist).item())
    return idx


def knn_interpolate_batched(
    values: torch.Tensor,
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    """Batched NN interpolate for concatenated sparse tensors with batch indices."""
    out_list = []
    B = int(target_batch.max().item()) + 1
    for b in range(B):
        sm = source_batch == b
        tm = target_batch == b
        out_list.append(
            knn_interpolate(values[sm], source_coords[sm], target_coords[tm])
        )
    return torch.cat(out_list, dim=0)


def build_query_shell_batched(
    partial_coord: torch.Tensor,
    partial_batch: torch.Tensor,
    num_query: int,
    radius: float,
    min_radius: float = QUERY_MIN_RADIUS,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One cylinder shell per batch item; returns (coords, batch_tensor)."""
    chunks = []
    batches = []
    B = int(partial_batch.max().item()) + 1
    for b in range(B):
        mask = partial_batch == b
        pc = partial_coord[mask]
        shell = build_query_shell(
            pc,
            num_query=num_query,
            radius=radius,
            min_radius=min_radius,
            device=partial_coord.device,
            generator=generator,
        )
        n = shell.shape[0]
        chunks.append(shell)
        batches.append(
            torch.full((n,), b, device=partial_coord.device, dtype=torch.long)
        )
    return torch.cat(chunks, dim=0), torch.cat(batches, dim=0)


def chamfer_symmetric_torch(
    pcd1: torch.Tensor,
    pcd2: torch.Tensor,
    max_points: int = 4096,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Symmetric squared Chamfer on point sets (subsampling for memory). Same spirit as eval scripts."""
    p1, p2 = pcd1.float(), pcd2.float()
    if p1.shape[0] > max_points:
        perm = torch.randperm(p1.shape[0], device=p1.device, generator=generator)[:max_points]
        p1 = p1[perm]
    if p2.shape[0] > max_points:
        perm = torch.randperm(p2.shape[0], device=p2.device, generator=generator)[:max_points]
        p2 = p2[perm]
    d = torch.cdist(p1, p2)
    c1 = d.min(dim=1)[0].pow(2).mean()
    c2 = d.min(dim=0)[0].pow(2).mean()
    return 0.5 * (c1 + c2)


class DiffusionScheduler:
    """
    Diffusion noise scheduler supporting multiple schedules.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",  # "linear", "cosine", "sigmoid"
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Initialize diffusion scheduler.
        
        Args:
            num_timesteps: Number of diffusion steps
            schedule: Type of noise schedule
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        
        # Generate beta schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_schedule(
                num_timesteps, beta_start, beta_end
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Compute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        
        # Precompute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
    
    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * np.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_schedule(
        self, 
        timesteps: int, 
        start: float = -3, 
        end: float = 3
    ) -> torch.Tensor:
        """Sigmoid schedule."""
        betas = torch.linspace(start, end, timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_start: Clean data (x_0)
            t: Timestep
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = \
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return (
            sqrt_alphas_cumprod_t * x_start + 
            sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def _to_device(self, device):
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recipm1_alphas = self.sqrt_recipm1_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)

    def p_sample_step(
        self,
        model: nn.Module,
        x_t_features: torch.Tensor,
        x_t_coords: torch.Tensor,
        t: int,
        condition: Dict[str, torch.Tensor],
        clip_denoised: bool = False
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t).
        Must be called with consecutive timesteps (t, t-1, ..., 0).
        """
        device = x_t_features.device
        self._to_device(device)

        # One shared timestep for all points; shape (N,) so time MLP matches point count in batch.
        N = x_t_features.shape[0]
        t_tensor = torch.full((N,), int(t), device=device, dtype=torch.long)
        pred_noise = model(x_t_features, x_t_coords, t_tensor, condition)
        pred_noise = torch.clamp(pred_noise, -PRED_NOISE_CLAMP, PRED_NOISE_CLAMP)

        # Predict x_0
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        pred_x0 = (
            x_t_features - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise
        ) / self.sqrt_alphas_cumprod[t]

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -50.0, 50.0)

        if t > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar) * pred_x0
                + torch.sqrt(self.alphas[t]) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * x_t_features
            )
            noise = torch.randn_like(x_t_features)
            return posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            return pred_x0


class SonataTransformerBlock(nn.Module):
    """
    Sonata-style transformer block for point cloud processing.
    
    Inspired by Point Transformer V3's grouped vector attention mechanism.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_groups: int = 4,
        num_neighbors: int = 16,
    ):
        """
        Initialize transformer block.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            num_groups: Number of groups for grouped vector attention
            num_neighbors: Number of neighbors for local attention
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_neighbors = num_neighbors
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert dim % num_groups == 0, "dim must be divisible by num_groups"
        
        # Grouped vector attention components
        self.group_dim = dim // num_groups
        
        # Query, Key, Value projections for each group
        self.q_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with grouped vector attention.
        
        Args:
            features: (N, dim) point features
            coords: (N, 3) point coordinates
            neighbors: (N, num_neighbors) neighbor indices (optional)
            
        Returns:
            (N, dim) transformed features
        """
        residual = features
        features = self.norm1(features)
        
        # Split features into groups
        group_features = torch.chunk(features, self.num_groups, dim=-1)
        
        # Process each group independently
        group_outputs = []
        for i, group_feat in enumerate(group_features):
            # Compute queries, keys, values
            q = self.q_proj[i](group_feat)  # (N, group_dim)
            k = self.k_proj[i](group_feat)  # (N, group_dim)
            v = self.v_proj[i](group_feat)  # (N, group_dim)
            
            # Find neighbors if not provided
            if neighbors is None:
                neighbors = self._find_neighbors(coords)
            
            # Local attention within neighbors
            attn_output = self._local_attention(q, k, v, coords, neighbors)
            group_outputs.append(attn_output)
        
        # Concatenate group outputs
        out = torch.cat(group_outputs, dim=-1)
        out = self.out_proj(out)
        out = out + residual
        
        # FFN
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = out + residual
        
        return out
    
    def _find_neighbors(self, coords: torch.Tensor) -> torch.Tensor:
        """Find k-nearest neighbors for each point using GPU."""
        N = coords.shape[0]
        k = min(self.num_neighbors + 1, N)
        chunk_size = 4096
        all_indices = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dists = torch.cdist(coords[start:end].unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
            _, idx = dists.topk(k, dim=-1, largest=False)
            all_indices.append(idx[:, 1:])
        result = torch.cat(all_indices, dim=0)
        if result.shape[1] < self.num_neighbors:
            pad = result[:, -1:].expand(-1, self.num_neighbors - result.shape[1])
            result = torch.cat([result, pad], dim=1)
        return result
    
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        neighbors: torch.Tensor
    ) -> torch.Tensor:
        """Compute local attention within neighbors."""
        N = q.shape[0]
        device = q.device
        
        # Gather neighbor features
        neighbor_k = k[neighbors]  # (N, num_neighbors, group_dim)
        neighbor_v = v[neighbors]  # (N, num_neighbors, group_dim)
        
        # Compute attention scores
        q_expanded = q.unsqueeze(1)  # (N, 1, group_dim)
        scores = torch.sum(q_expanded * neighbor_k, dim=-1) / np.sqrt(self.group_dim)
        attn_weights = F.softmax(scores, dim=-1)  # (N, num_neighbors)
        
        # Apply attention
        out = torch.sum(attn_weights.unsqueeze(-1) * neighbor_v, dim=1)  # (N, group_dim)
        
        return out


class PointwiseDiffusionBlock(nn.Module):
    """
    Point-wise diffusion block using Sonata-style transformer.
    
    Processes each point's local neighborhood with transformer attention,
    replacing sparse convolutions with Sonata's grouped vector attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int = 128,
        condition_dim: int = 256,
        num_neighbors: int = 16,
        num_heads: int = 8,
        num_groups: int = 4,
        conditioning_mode: str = "concat",
        conditioning_scale: float = 1.0,
    ):
        """
        Initialize point-wise diffusion block.
        
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            condition_dim: Conditional feature dimension
            num_neighbors: Number of neighbors for local processing
            num_heads: Number of attention heads
            num_groups: Number of groups for grouped vector attention
            conditioning_mode: How to fuse encoder features (concat | additive | film).
            conditioning_scale: Strength for additive / FiLM paths (concat ignores scale).
        """
        super().__init__()
        
        self.num_neighbors = num_neighbors
        mode = str(conditioning_mode).lower()
        if mode not in ("concat", "additive", "film"):
            raise ValueError(f"conditioning_mode must be concat|additive|film, got {conditioning_mode!r}")
        self.conditioning_mode = mode
        self.conditioning_scale = float(conditioning_scale)
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Condition projection + fusion: concat makes cond a hard bottleneck (additive cond was easy to ignore).
        self.condition_proj = nn.Linear(condition_dim, in_channels)
        self.cond_fuse = nn.Linear(in_channels * 2, in_channels)

        # Sonata-style transformer block
        self.transformer = SonataTransformerBlock(
            dim=in_channels,
            num_heads=num_heads,
            num_groups=num_groups,
            num_neighbors=num_neighbors
        )
        
        # Output projection
        if in_channels != out_channels:
            self.out_proj = nn.Linear(in_channels, out_channels)
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        time_embed: torch.Tensor,
        condition: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (N, in_channels) point features
            coords: (N, 3) point coordinates
            time_embed: (batch_size, time_embed_dim) time step embedding
            condition: (N, condition_dim) conditional features from encoder
            neighbors: (N, num_neighbors) neighbor indices (optional)
            
        Returns:
            (N, out_channels) processed features
        """
        # time_embed is (N, time_embed_dim): one diffusion time per point (batched scenes).
        time_feat = self.time_mlp(time_embed)
        x_feat = features + time_feat

        cond_feat = self.condition_proj(condition)
        cond_feat = torch.clamp(cond_feat, -10.0, 10.0)

        if self.conditioning_mode == "concat":
            x_feat = self.cond_fuse(torch.cat([x_feat, cond_feat], dim=-1))
        elif self.conditioning_mode == "additive":
            x_feat = x_feat + self.conditioning_scale * cond_feat
        else:  # film
            scale = torch.tanh(cond_feat)
            x_feat = x_feat * (1.0 + self.conditioning_scale * scale)

        if self.training and torch.rand(1).item() < 0.01:
            print("Cond influence check:", cond_feat.abs().mean().item())

        # Transformer processing
        x_transformed = self.transformer(x_feat, coords, neighbors)
        
        # Output projection
        out = self.out_proj(x_transformed)
        
        return out


class DenoisingNetwork(nn.Module):
    """
    U-Net style denoising network using Sonata transformer blocks.
    
    Predicts noise at each diffusion step, conditioned on:
    - Partial input scan
    - Sonata encoder features
    - Timestep
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # xyz coordinates
        condition_dim: int = 256,
        hidden_dims: list = [64, 128, 256, 512],
        time_embed_dim: int = 128,
        num_neighbors: int = 16,
        conditioning_mode: str = "concat",
        conditioning_scale: float = 1.0,
    ):
        """
        Initialize denoising network.
        
        Args:
            in_channels: Input point feature dimension
            condition_dim: Conditional feature dimension from Sonata
            hidden_dims: Hidden dimensions for U-Net levels
            time_embed_dim: Time embedding dimension
            num_neighbors: Neighbors for local processing
            conditioning_mode: Passed to each PointwiseDiffusionBlock (concat | additive | film).
            conditioning_scale: Strength for additive / FiLM blocks.
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.hidden_dims = hidden_dims
        self.conditioning_mode = str(conditioning_mode).lower()
        self.conditioning_scale = float(conditioning_scale)
        
        # Sinusoidal time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dims[0])
        
        # Encoder blocks (downsampling)
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # Diffusion block
            self.encoder_blocks.append(
                PointwiseDiffusionBlock(
                    hidden_dims[i], hidden_dims[i],
                    time_embed_dim, condition_dim, num_neighbors,
                    conditioning_mode=self.conditioning_mode,
                    conditioning_scale=self.conditioning_scale,
                )
            )
        

        # Projections between levels (dimension transitions)
        self.level_up = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.level_up.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Bottleneck
        self.bottleneck = PointwiseDiffusionBlock(
            hidden_dims[-1], hidden_dims[-1],
            time_embed_dim, condition_dim, num_neighbors,
            conditioning_mode=self.conditioning_mode,
            conditioning_scale=self.conditioning_scale,
        )
        
        # Decoder blocks (upsampling)
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Diffusion block with skip connection
            self.decoder_blocks.append(
                PointwiseDiffusionBlock(
                    hidden_dims[i - 1] + hidden_dims[i], hidden_dims[i - 1],
                    time_embed_dim, condition_dim, num_neighbors,
                    conditioning_mode=self.conditioning_mode,
                    conditioning_scale=self.conditioning_scale,
                )
            )
        
        # Output projection (predict noise)
        self.output_proj = nn.Linear(hidden_dims[0], in_channels)
    
    def _downsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample points using farthest point sampling."""
        N = features.shape[0]
        if N <= target_num:
            return features, coords

        indices = farthest_point_sample(coords, target_num)
        return features[indices], coords[indices]
    
    def _upsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_coords: torch.Tensor
    ) -> torch.Tensor:
        """Upsample features to target coordinates using GPU nearest neighbor."""
        chunk_size = 4096
        all_indices = []
        for start in range(0, target_coords.shape[0], chunk_size):
            end = min(start + chunk_size, target_coords.shape[0])
            dists = torch.cdist(target_coords[start:end].unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
            all_indices.append(dists.argmin(dim=-1))
        indices = torch.cat(all_indices, dim=0)
        return features[indices]
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        condition: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict noise for denoising step.
        
        Args:
            features: (N, in_channels) noisy point features
            coords: (N, 3) point coordinates
            timestep: Current timestep (batch_size,)
            condition: Conditional features from Sonata encoder
            
        Returns:
            (N, in_channels) predicted noise
        """
        # Time embedding: expect one timestep per point (N,) for mixed batch sizes.
        t_embed = self.time_embedding(timestep.reshape(-1).long())
        
        # Get conditional features
        cond_feat = condition['features']
        
        # Input projection
        x = self.input_proj(features)
        x_coords = coords
        x_cond = cond_feat
        
        # Encoder path
        skip_features = []
        skip_coords = []
        skip_conds = []
        
        for i, enc_block in enumerate(self.encoder_blocks):
            # Process with transformer at current level
            x = enc_block(x, x_coords, t_embed, x_cond)

            # Save skip connection
            skip_features.append(x)
            skip_coords.append(x_coords)
            skip_conds.append(x_cond)

            # Downsample and project to next level dimension
            target_num = x.shape[0] // 2
            N = x.shape[0]
            if N > target_num:
                indices = farthest_point_sample(x_coords, target_num)
                x = x[indices]
                x_coords = x_coords[indices]
                x_cond = x_cond[indices]
                t_embed = t_embed[indices]
            # Project to next level dimension
            x = self.level_up[i](x)

        # Bottleneck
        x = self.bottleneck(x, x_coords, t_embed, x_cond)
        
        # Decoder path
        for i, dec_block in enumerate(self.decoder_blocks):
            # Upsample to match skip connection
            skip_feat = skip_features[-(i+1)]
            skip_coord = skip_coords[-(i+1)]
            skip_cond = skip_conds[-(i+1)]
            
            x = self._upsample_points(x, x_coords, skip_coord)
            t_embed = self._upsample_points(t_embed, x_coords, skip_coord)
            x_coords = skip_coord
            x_cond = skip_cond

            # Concatenate skip connection
            x = torch.cat([x, skip_feat], dim=-1)

            # Process with transformer
            x = dec_block(x, x_coords, t_embed, x_cond)
        
        # Output projection
        noise_pred = self.output_proj(x)
        
        return noise_pred


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.
        
        Args:
            timesteps: (batch_size,) timestep values
            
        Returns:
            (batch_size, embed_dim) embeddings
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2
        
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([
            torch.sin(embeddings), torch.cos(embeddings)
        ], dim=-1)
        
        return embeddings



def knn_interpolate(features, source_coords, target_coords):
    """Map features from source points to target points via nearest neighbor (GPU-friendly)."""
    chunk = 4096
    indices_list = []
    dtype = source_coords.dtype
    for s in range(0, target_coords.shape[0], chunk):
        e = min(s + chunk, target_coords.shape[0])
        d = torch.cdist(target_coords[s:e].float(), source_coords.float())
        indices_list.append(d.argmin(dim=1))
    indices = torch.cat(indices_list, dim=0).to(dtype=torch.long, device=features.device)
    return features[indices]


class SceneCompletionDiffusion(nn.Module):
    """
    Complete diffusion model for semantic scene completion.
    
    Combines:
    - Sonata encoder for conditional features
    - Diffusion scheduler for noise schedule
    - Denoising network for reverse process
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        condition_extractor: nn.Module,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        denoising_steps: int = 50,
        chamfer_max_points: int = 4096,
        num_query_extra: Optional[int] = None,
        query_radius: Optional[float] = None,
        query_min_radius: float = QUERY_MIN_RADIUS,
        chamfer_lambda: float = 0.1,
        center_lambda: float = 0.02,
        conditioning_mode: str = "concat",
        conditioning_scale: float = 1.0,
        condition_dropout_prob: float = 0.1,
    ):
        """
        Initialize complete diffusion model.
        
        Args:
            encoder: Sonata encoder
            condition_extractor: Conditional feature extractor
            num_timesteps: Total diffusion steps
            schedule: Noise schedule type
            denoising_steps: Steps for inference
            chamfer_max_points: Subsample cap for Chamfer in loss (memory)
            num_query_extra: Random query count per sample (plus partial voxels)
            query_radius: Cylinder radius for extra queries (sensor at origin)
            query_min_radius: Inner xy radius excluded for synthetic queries (blind zone)
            chamfer_lambda: Weight for Chamfer on predicted x0
            center_lambda: Weight for global centroid alignment (anchor prior)
            conditioning_mode: Denoiser fusion of Sonata features (concat | additive | film).
            conditioning_scale: Strength for additive / FiLM (ignored for concat).
            condition_dropout_prob: Training-only: zero cond with this prob (CFG-style; 0 = off).
        """
        super().__init__()
        
        self.encoder = encoder
        self.condition_extractor = condition_extractor
        self.denoising_steps = denoising_steps
        self.chamfer_max_points = chamfer_max_points
        self.num_query_extra = (
            int(num_query_extra) if num_query_extra is not None else DEFAULT_TRAIN_MAX_POINTS
        )
        self.query_radius = (
            float(query_radius) if query_radius is not None else float(QUERY_MAX_RADIUS)
        )
        self.query_min_radius = float(query_min_radius)
        self.chamfer_lambda = chamfer_lambda
        self.center_lambda = center_lambda
        self.conditioning_mode = str(conditioning_mode).lower()
        self.conditioning_scale = float(conditioning_scale)
        self.condition_dropout_prob = float(condition_dropout_prob)
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule=schedule
        )
        
        # Denoising network: in_channels=3 => noise lives in coordinate space (x, y, z).
        self.denoiser = DenoisingNetwork(
            in_channels=3,
            condition_dim=condition_extractor.out_dim,
            hidden_dims=[64, 128, 256, 512],
            time_embed_dim=128,
            conditioning_mode=self.conditioning_mode,
            conditioning_scale=self.conditioning_scale,
        )
    
    def forward(
        self,
        partial_scan: Dict[str, torch.Tensor],
        complete_scan: torch.Tensor,
        complete_batch: torch.Tensor = None,
        return_loss: bool = True,
        condition_features: torch.Tensor = None,
        log_metrics: bool = True,
        log_conditioning: bool = False,
        conditioning_loss_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Training: diffuse NN-interpolated GT onto a fixed query shell; denoiser sees
        noisy coords x_t and static query layout coords for graph geometry.

        ``conditioning_loss_weight``: if > 0 in training, runs a second denoiser pass on the
        same ``x_t``, timesteps, and noise with zero conditioning features. Noise loss is
        ``loss_cond + weight * loss_zero`` where ``loss_*`` are MSE to ground-truth noise.
        Logged ``conditioning_advantage`` is ``loss_zero - loss_cond`` (monitoring only).
        """
        partial_coord = partial_scan["coord"]
        partial_batch = partial_scan.get("batch")
        if partial_batch is None:
            partial_batch = torch.zeros(
                partial_coord.shape[0], dtype=torch.long, device=partial_coord.device
            )

        query_coords, query_batch = build_query_shell_batched(
            partial_coord,
            partial_batch,
            num_query=self.num_query_extra,
            radius=self.query_radius,
            min_radius=self.query_min_radius,
        )

        # GT targets on query points: NN in metric space from full GT cloud (per batch).
        gt_x0 = knn_interpolate_batched(
            complete_scan,
            complete_scan,
            query_coords,
            complete_batch,
            query_batch,
        )

        if condition_features is not None:
            cond_mapped = knn_interpolate_batched(
                condition_features,
                complete_scan,
                query_coords,
                complete_batch,
                query_batch,
            )
        else:
            cond_features, _ = self.condition_extractor(partial_scan)
            cond_mapped = knn_interpolate_batched(
                cond_features,
                partial_coord,
                query_coords,
                partial_batch,
                query_batch,
            )

        # Training-only: randomly drop cond so the denoiser must handle both regimes (CFG-style).
        # Skip dropout when computing the cond-vs-zero auxiliary (needs a clean cond path).
        if (
            self.training
            and self.condition_dropout_prob > 0.0
            and float(conditioning_loss_weight) <= 0.0
        ):
            if torch.rand(1).item() < self.condition_dropout_prob:
                cond_mapped = torch.zeros_like(cond_mapped)

        if log_conditioning:
            print(
                "COND stats (train, cond_mapped):",
                cond_mapped.mean().item(),
                cond_mapped.std().item(),
            )

        if complete_batch is not None:
            batch_size = int(complete_batch.max().item()) + 1
        else:
            batch_size = 1

        t_hi = self.scheduler.num_timesteps
        t_lo = TRAIN_TIMESTEP_MIN if t_hi > TRAIN_TIMESTEP_MIN else 0
        t = torch.randint(
            t_lo,
            t_hi,
            (batch_size,),
            device=query_coords.device,
        )
        t_per_point = t[query_batch]

        noise = torch.randn_like(gt_x0)
        dev = query_coords.device
        sa = self.scheduler.sqrt_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
        som = self.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
        x_t = sa * gt_x0 + som * noise

        timestep_per_point = t_per_point.long()
        pred_noise = self.denoiser(
            x_t,
            query_coords,
            timestep_per_point,
            {"features": cond_mapped},
        )
        pred_noise = torch.clamp(pred_noise, -PRED_NOISE_CLAMP, PRED_NOISE_CLAMP)

        use_cond_aux = (
            self.training
            and float(conditioning_loss_weight) > 0.0
            and return_loss
        )
        if use_cond_aux:
            pred_noise_zero = self.denoiser(
                x_t,
                query_coords,
                timestep_per_point,
                {"features": torch.zeros_like(cond_mapped)},
            )
            pred_noise_zero = torch.clamp(
                pred_noise_zero, -PRED_NOISE_CLAMP, PRED_NOISE_CLAMP
            )
        else:
            pred_noise_zero = None

        som_ab = self.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
        sqrt_ab = self.scheduler.sqrt_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1).clamp(min=1e-8)
        pred_x0 = (x_t - som_ab * pred_noise) / sqrt_ab

        if not return_loss:
            return {"pred_noise": pred_noise, "pred_x0": pred_x0}

        loss_cond = F.mse_loss(pred_noise, noise)
        if use_cond_aux and pred_noise_zero is not None:
            loss_zero = F.mse_loss(pred_noise_zero, noise)
            lam = float(conditioning_loss_weight)
            noise_loss = loss_cond + lam * loss_zero
            conditioning_advantage = loss_zero - loss_cond
        else:
            loss_zero = None
            conditioning_advantage = None
            noise_loss = loss_cond

        # Chamfer-only clamp: limits exploding ||pred_x0|| when sqrt(alpha_bar) is tiny (no loss mask:
        # random queries already avoid r_xy < min_radius, while partial voxels may still lie inside).
        Rc = self.query_radius
        pred_x0_ch = torch.clamp(pred_x0, -Rc, Rc)
        loss_chamfer = chamfer_symmetric_torch(
            pred_x0_ch,
            gt_x0,
            max_points=self.chamfer_max_points,
        )
        pm = pred_x0_ch.mean(dim=0)
        gm = gt_x0.mean(dim=0)
        loss_center = torch.sum((pm - gm) ** 2)

        main_loss = (
            loss_cond
            + self.chamfer_lambda * loss_chamfer
            + self.center_lambda * loss_center
        )
        loss = (
            noise_loss
            + self.chamfer_lambda * loss_chamfer
            + self.center_lambda * loss_center
        )

        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "main_loss": main_loss.detach(),
            "loss_mse": loss_cond.detach(),
            "loss_chamfer": loss_chamfer.detach(),
            "loss_center": loss_center.detach(),
            "loss_cond": loss_cond.detach(),
            "loss_zero": loss_zero.detach() if loss_zero is not None else None,
            "conditioning_advantage": (
                conditioning_advantage.detach()
                if conditioning_advantage is not None
                else None
            ),
            "pred_noise": pred_noise,
            "pred_x0": pred_x0,
        }
        if log_metrics:
            out["pred_x0_mean"] = pred_x0.mean().detach()
            out["pred_x0_std"] = pred_x0.std().detach()
        return out
    
    @torch.no_grad()
    def complete_scene(
        self,
        partial_scan: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        use_query_points: bool = True,
        shell_coords: Optional[torch.Tensor] = None,
        num_query_extra: Optional[int] = None,
        query_radius: Optional[float] = None,
        anchor_alpha: float = 1.0,
        anchor_start_step: Optional[int] = None,
        denoise_snapshot_iters: Optional[Set[int]] = None,
        on_denoise_snapshot: Optional[Callable[[int, torch.Tensor], None]] = None,
        target_coords: Optional[torch.Tensor] = None,
        diagnostics: bool = False,
        zero_condition: bool = False,
        noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference uses the same query shell as training when ``use_query_points``.
        ``targetCoords`` / ``shell_coords`` override the learned layout if provided.

        ``anchor_start_step``: if set, anchor applies only on DDIM sub-steps with
        ``t_idx <= anchor_start_step`` (``t_idx`` counts down from ``num_steps-1`` to ``0``).
        When ``t_idx > anchor_start_step``, partial rows are left to the denoiser (softer LiDAR lock).

        Diagnostic-only (no effect on training): ``diagnostics``, ``zero_condition``,
        ``noise_seed`` — see ``scripts/diagnose_completion.py``.
        """
        if num_steps is None:
            num_steps = self.denoising_steps

        device = next(self.parameters()).device
        for k in partial_scan:
            if isinstance(partial_scan[k], torch.Tensor):
                partial_scan[k] = partial_scan[k].to(device)

        partial_coord = partial_scan["coord"]
        cond_features, _ = self.condition_extractor(partial_scan)

        if target_coords is not None:
            coords = target_coords.to(device)
        elif shell_coords is not None:
            coords = shell_coords.to(device)
        elif use_query_points:
            n_q = self.num_query_extra if num_query_extra is None else int(num_query_extra)
            rad = self.query_radius if query_radius is None else float(query_radius)
            coords = build_query_shell(
                partial_coord,
                num_query=n_q,
                radius=rad,
                min_radius=self.query_min_radius,
                device=device,
            )
        else:
            coords = partial_coord

        cond_features = knn_interpolate(
            cond_features, partial_scan["coord"], coords
        )

        if zero_condition:
            cond_features = torch.zeros_like(cond_features)

        if diagnostics:
            print(
                "COND stats:",
                cond_features.mean().item(),
                cond_features.std().item(),
            )
            n_q = int(coords.shape[0]) - int(partial_coord.shape[0])
            r_eff = (
                float(query_radius)
                if query_radius is not None
                else float(self.query_radius)
            )
            print(
                "Shell info: partial:",
                partial_coord.shape[0],
                "total:",
                coords.shape[0],
                "query:",
                n_q,
            )
            print("Radius:", r_eff, "| Min radius:", float(self.query_min_radius))

        if noise_seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(noise_seed))
            x_t = torch.randn(
                coords.shape, dtype=coords.dtype, device=device, generator=g
            )
        else:
            x_t = torch.randn_like(coords)

        n_partial = int(partial_coord.shape[0])
        x_init = x_t.clone() if diagnostics else None

        if denoise_snapshot_iters is not None and on_denoise_snapshot is not None and 0 in denoise_snapshot_iters:
            on_denoise_snapshot(0, x_t)

        for step_i, t in enumerate(range(num_steps - 1, -1, -1)):
            x_t = self.scheduler.p_sample_step(
                self.denoiser, x_t, coords, t, {"features": cond_features}
            )
            apply_anchor = n_partial > 0 and float(anchor_alpha) != 0.0
            if apply_anchor and anchor_start_step is not None:
                if t > int(anchor_start_step):
                    apply_anchor = False
            if apply_anchor:
                aa = float(anchor_alpha)
                if aa < 1.0:
                    x_t[:n_partial] = (
                        aa * partial_coord + (1.0 - aa) * x_t[:n_partial]
                    )
                else:
                    x_t[:n_partial] = partial_coord
            if (
                denoise_snapshot_iters is not None
                and on_denoise_snapshot is not None
                and (step_i + 1) in denoise_snapshot_iters
            ):
                on_denoise_snapshot(step_i + 1, x_t)

        if diagnostics and x_init is not None:
            print(
                "Diffusion movement:",
                (x_t - x_init).abs().mean().item(),
            )

        return x_t


if __name__ == "__main__":
    print("Testing Diffusion Module...")
    
    # Test scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000, schedule="cosine")
    print(f"Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    
    # Test diffusion block
    block = PointwiseDiffusionBlock(64, 128, 128, 256).cuda()
    print(f"\nDiffusion block created: {block}")