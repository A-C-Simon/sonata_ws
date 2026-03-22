"""
Diffusion Module for Semantic Scene Completion

Implements the diffusion process for point cloud completion,
following LiDiff's point-wise local approach with Sonata encoder conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


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
    
    def p_sample_step(
        self,
        model: nn.Module,
        x_t_features: torch.Tensor,
        x_t_coords: torch.Tensor,
        t: int,
        condition: Dict[str, torch.Tensor],
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x_t_features: (N, 3) Noisy features at timestep t
            x_t_coords: (N, 3) Point coordinates
            t: Current timestep
            condition: Conditional information
            clip_denoised: Clip denoised output
            
        Returns:
            (N, 3) Denoised features at timestep t-1
        """
        N = x_t_features.shape[0]
        t_per_point = torch.full((N,), t, device=x_t_features.device, dtype=torch.long)

        # Model predicts residual delta; x0 = x_t + delta (more stable)
        pred_delta = model(x_t_features, x_t_coords, t_per_point, condition)
        pred_x0 = x_t_features + pred_delta

        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]
        # Derive pred_noise for posterior (needed for DDPM step)
        pred_noise = (
            x_t_features - self.sqrt_alphas_cumprod[t] * pred_x0
        ) / self.sqrt_one_minus_alphas_cumprod[t].clamp(min=1e-8)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t_features)
            posterior_variance_t = self.posterior_variance[t]
            
            # x_{t-1} = posterior_mean + sqrt(posterior_variance) * noise
            posterior_mean = (
                self.sqrt_alphas_cumprod[t - 1] * pred_x0 +
                torch.sqrt(1 - alpha_t_prev - posterior_variance_t) * pred_noise
            )
            
            x_t_prev = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_t_prev = pred_x0
        
        return x_t_prev


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
        """KNN using matmul (||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b) to avoid large cdist."""
        N = coords.shape[0]
        k = min(self.num_neighbors + 1, N)
        if k <= 1:
            return torch.zeros(N, self.num_neighbors, dtype=torch.long, device=coords.device)
        sq = (coords ** 2).sum(dim=1)
        chunk_size = 4096
        all_indices = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = coords[start:end]
            dist_sq = sq[start:end].unsqueeze(1) + sq.unsqueeze(0) - 2 * (chunk @ coords.T)
            dist_sq = dist_sq.clamp(min=0)
            _, idx = dist_sq.topk(k, dim=-1, largest=False)
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
        """
        super().__init__()
        
        self.num_neighbors = num_neighbors
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, in_channels)
        
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
            time_embed: (N, time_embed_dim) per-point time embedding
            condition: (N, condition_dim) conditional features from encoder
            neighbors: (N, num_neighbors) neighbor indices (optional)

        Returns:
            (N, out_channels) processed features
        """
        # Per-point time embedding: (N, embed_dim) -> (N, in_channels)
        time_feat = self.time_mlp(time_embed)
        x_feat = features + time_feat
        
        # Apply condition
        cond_feat = self.condition_proj(condition)
        x_feat = x_feat + cond_feat
        
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
    ):
        """
        Initialize denoising network.
        
        Args:
            in_channels: Input point feature dimension
            condition_dim: Conditional feature dimension from Sonata
            hidden_dims: Hidden dimensions for U-Net levels
            time_embed_dim: Time embedding dimension
            num_neighbors: Neighbors for local processing
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.hidden_dims = hidden_dims
        
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
                    time_embed_dim, condition_dim, num_neighbors
                )
            )
        

        # Projections between levels (dimension transitions)
        self.level_up = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.level_up.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Bottleneck
        self.bottleneck = PointwiseDiffusionBlock(
            hidden_dims[-1], hidden_dims[-1],
            time_embed_dim, condition_dim, num_neighbors
        )
        
        # Decoder blocks (upsampling)
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Diffusion block with skip connection
            self.decoder_blocks.append(
                PointwiseDiffusionBlock(
                    hidden_dims[i - 1] + hidden_dims[i], hidden_dims[i - 1],
                    time_embed_dim, condition_dim, num_neighbors
                )
            )
        
        # Output projection (predict noise)
        self.output_proj = nn.Linear(hidden_dims[0], in_channels)
    
    def _fps_indices(self, coords: torch.Tensor, target_num: int) -> torch.Tensor:
        """Farthest Point Sampling: indices that preserve geometry."""
        N = coords.shape[0]
        if N <= target_num:
            return torch.arange(N, device=coords.device, dtype=torch.long)
        idx = torch.zeros(target_num, dtype=torch.long, device=coords.device)
        idx[0] = torch.randint(0, N, (1,), device=coords.device).item()
        dists = torch.full((N,), float("inf"), device=coords.device, dtype=coords.dtype)
        chunk_sz = 4096
        for i in range(1, target_num):
            center = coords[idx[i - 1]]
            for start in range(0, N, chunk_sz):
                end = min(start + chunk_sz, N)
                d = torch.sum((coords[start:end] - center) ** 2, dim=-1)
                dists[start:end] = torch.minimum(dists[start:end], d)
            idx[i] = dists.argmax().item()
        return idx

    def _downsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample using FPS to preserve geometry."""
        N = features.shape[0]
        if N <= target_num:
            return features, coords
        indices = self._fps_indices(coords, target_num)
        return features[indices], coords[indices]
    
    def _upsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_coords: torch.Tensor
    ) -> torch.Tensor:
        """Upsample features via nearest neighbor (matmul-based KNN)."""
        n_tgt = target_coords.shape[0]
        n_src = coords.shape[0]
        if n_src == 0:
            return torch.zeros(n_tgt, features.shape[-1], device=features.device, dtype=features.dtype)
        sq_src = (coords ** 2).sum(dim=1)
        chunk_size = 4096
        all_idx = []
        for start in range(0, n_tgt, chunk_size):
            end = min(start + chunk_size, n_tgt)
            tgt = target_coords[start:end]
            sq_tgt = (tgt ** 2).sum(dim=1)
            dist_sq = sq_tgt.unsqueeze(1) + sq_src.unsqueeze(0) - 2 * (tgt @ coords.T)
            all_idx.append(dist_sq.argmin(dim=-1))
        indices = torch.cat(all_idx, dim=0)
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
            timestep: (N,) per-point timesteps (no broadcast/repeat)
            condition: Conditional features from Sonata encoder

        Returns:
            (N, in_channels) predicted noise
        """
        # Per-point time embedding: (N,) -> (N, embed_dim)
        t_embed = self.time_embedding(timestep)
        
        # Get conditional features
        cond_feat = condition['features']
        
        # Input projection
        x = self.input_proj(features)
        x_coords = coords
        x_cond = cond_feat
        
        # Encoder path (downsample t_embed with coords)
        skip_features = []
        skip_coords = []
        skip_conds = []

        for i, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x, x_coords, t_embed, x_cond)
            skip_features.append(x)
            skip_coords.append(x_coords)
            skip_conds.append(x_cond)

            target_num = max(1, x.shape[0] // 2)
            if x.shape[0] > target_num:
                indices = self._fps_indices(x_coords, target_num)
                x = x[indices]
                x_coords = x_coords[indices]
                x_cond = x_cond[indices]
                t_embed = t_embed[indices]
            x = self.level_up[i](x)

        # Bottleneck
        x = self.bottleneck(x, x_coords, t_embed, x_cond)

        # Decoder path (upsample t_embed to match skip)
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_feat = skip_features[-(i+1)]
            skip_coord = skip_coords[-(i+1)]
            skip_cond = skip_conds[-(i+1)]
            t_embed = self._upsample_points(t_embed, x_coords, skip_coord)

            x = self._upsample_points(x, x_coords, skip_coord)
            x_coords = skip_coord
            x_cond = skip_cond

            x = torch.cat([x, skip_feat], dim=-1)
            x = dec_block(x, x_coords, t_embed, x_cond)
        
        # Output projection
        noise_pred = self.output_proj(x)
        
        return noise_pred


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps (per-point)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.

        Args:
            timesteps: (N,) per-point timestep values

        Returns:
            (N, embed_dim) embeddings — one per point, no repeat
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2

        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device, dtype=timesteps.dtype) * -embeddings
        )
        # timesteps: (N,) -> (N, 1) * (1, half_dim) -> (N, half_dim)
        emb = timesteps.float().unsqueeze(-1) * embeddings.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb



def weighted_interpolate(
    features: torch.Tensor,
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    k: int = 4,
    eps: float = 1e-8,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Map features from source to target via inverse-distance weighted interpolation."""
    device = features.device
    feat_dim = features.shape[-1]
    n_target = target_coords.shape[0]
    n_src = source_coords.shape[0]
    k = min(k, n_src)
    if k <= 0 or n_src == 0:
        return torch.zeros(n_target, feat_dim, device=device, dtype=features.dtype)

    out = []
    for start in range(0, n_target, chunk_size):
        end = min(start + chunk_size, n_target)
        tgt = target_coords[start:end]
        diff = tgt.unsqueeze(1) - source_coords.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1).clamp(min=eps)
        dist = dist_sq.sqrt()
        weights, idx = torch.topk(-dist, k, dim=1)
        weights = (-weights).clamp(min=eps)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        gathered = features[idx]
        interp = (weights.unsqueeze(-1) * gathered).sum(dim=1)
        out.append(interp)
    return torch.cat(out, dim=0)


def knn_interpolate(features, source_coords, target_coords):
    """Legacy alias; use weighted_interpolate for better quality."""
    return weighted_interpolate(features, source_coords, target_coords, k=4)


def chamfer_distance_training(
    pred: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Symmetric mean squared Chamfer between two point clouds.
    Chunked to avoid OOM on large N, M (same pattern as refinement_net).
    """
    def min_dist_sq_dir(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mins = []
        for i in range(0, a.shape[0], chunk_size):
            chunk = a[i : i + chunk_size]
            diff = chunk.unsqueeze(1) - b.unsqueeze(0)
            dist = (diff ** 2).sum(-1)
            mins.append(dist.min(1)[0])
        return torch.cat(mins, dim=0)

    min_p2t = min_dist_sq_dir(pred, target)
    min_t2p = min_dist_sq_dir(target, pred)
    return (min_p2t.mean() + min_t2p.mean()) / 2


# Aligned with map_from_scans_boost: output_radius=20m, scan_ego_min_range=3.5m
QUERY_MAX_RADIUS = 20.0
QUERY_MIN_RADIUS = 3.5
# Match SemanticKITTI default max_points (partial + query ~= full scene cap)
DEFAULT_TARGET_TOTAL_POINTS = 20000


def _build_cylindrical_grid(
    min_radius: float,
    max_radius: float,
    z_min: float,
    z_max: float,
    voxel_size: float,
) -> np.ndarray:
    """Dense cylindrical candidate points (numpy)."""
    r = np.arange(min_radius, max_radius + voxel_size * 0.5, voxel_size)
    theta = np.arange(0, 2 * np.pi, max(voxel_size / max_radius, 1e-6))
    z = np.arange(z_min, z_max + voxel_size * 0.5, voxel_size)
    if r.size == 0 or theta.size == 0 or z.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rr, tt, zz = np.meshgrid(r, theta, z, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    return np.stack([x, y, zz], axis=-1).reshape(-1, 3)


def _uniform_frustum_fill(
    n_needed: int,
    partial_np: np.ndarray,
    min_radius: float,
    max_radius: float,
    z_min: float,
    z_max: float,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points in cylindrical frustum, away from partial (rejection)."""
    if n_needed <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    tree = cKDTree(partial_np) if partial_np.shape[0] > 0 else None
    out: list[np.ndarray] = []
    max_tries = max(10000, n_needed * 200)
    tries = 0
    while len(out) < n_needed and tries < max_tries:
        tries += 1
        r = np.sqrt(rng.random() * (max_radius**2 - min_radius**2) + min_radius**2)
        th = rng.random() * 2 * np.pi
        z = z_min + rng.random() * (z_max - z_min)
        p = np.array([r * np.cos(th), r * np.sin(th), z], dtype=np.float64)
        if tree is not None:
            d, _ = tree.query(p, k=1)
            if d < min_sep:
                continue
        if out:
            arr = np.stack(out, axis=0)
            if np.linalg.norm(arr - p, axis=1).min() < min_sep * 0.4:
                continue
        out.append(p)
    # Shortfall: jitter around partial / radial directions
    short = n_needed - len(out)
    if short > 0 and partial_np.shape[0] > 0:
        for _ in range(short):
            i = int(rng.integers(0, partial_np.shape[0]))
            direction = partial_np[i].copy()
            nrm = np.linalg.norm(direction[:2])
            if nrm < 1e-3:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / (np.linalg.norm(direction) + 1e-9)
            scale = rng.uniform(min_radius, max_radius)
            p = direction * scale
            p[2] = float(
                np.clip(
                    partial_np[i, 2] + rng.normal(0.0, 0.3),
                    z_min,
                    z_max,
                )
            )
            out.append(p.astype(np.float64))
    elif short > 0:
        for _ in range(short):
            r = np.sqrt(rng.random() * (max_radius**2 - min_radius**2) + min_radius**2)
            th = rng.random() * 2 * np.pi
            z = z_min + rng.random() * (z_max - z_min)
            out.append(
                np.array([r * np.cos(th), r * np.sin(th), z], dtype=np.float64)
            )
    return np.stack(out, axis=0)[:n_needed]


def generate_query_points(
    partial_coord: torch.Tensor,
    max_radius: float = QUERY_MAX_RADIUS,
    min_radius: float = QUERY_MIN_RADIUS,
    voxel_size: float = 0.15,
    target_total_points: int = DEFAULT_TARGET_TOTAL_POINTS,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """
    Add exactly the number of query points needed to reach target_total_points.

    num_query = max(0, target_total_points - len(partial)), aligned with training
    where complete scenes are capped (e.g. SemanticKITTI max_points=20000).

    Args:
        partial_coord: (N, 3) partial scan coords (centered at origin)
        max_radius: max distance from origin (m)
        min_radius: min distance from origin (m)
        voxel_size: initial grid step (m); refined if not enough candidates
        target_total_points: desired total points (partial + query)
        rng: optional RNG for subsampling / fill

    Returns:
        (target_total_points, 3) if N_partial < target; else partial only
        (i.e. partial_coord unchanged when already at or above target)
    """
    device = partial_coord.device
    rng = rng or np.random.default_rng()
    partial_np = partial_coord.detach().cpu().numpy().astype(np.float64)
    n_partial = int(partial_coord.shape[0])
    num_query_needed = max(0, target_total_points - n_partial)
    if num_query_needed == 0:
        return partial_coord

    z_min = float(np.clip(partial_np[:, 2].min() - 2.0, -max_radius, max_radius))
    z_max = float(np.clip(partial_np[:, 2].max() + 2.0, -max_radius, max_radius))

    v = float(voxel_size)
    v_min = 0.05
    grid = np.zeros((0, 3), dtype=np.float64)
    for _ in range(10):
        grid = _build_cylindrical_grid(
            min_radius, max_radius, z_min, z_max, v
        )
        if partial_np.shape[0] > 0 and grid.shape[0] > 0:
            tree = cKDTree(partial_np)
            nn_dist, _ = tree.query(grid, k=1)
            grid = grid[nn_dist > v * 0.6]
        if grid.shape[0] >= num_query_needed:
            break
        v = max(v * 0.82, v_min)

    if grid.shape[0] >= num_query_needed:
        idx = rng.choice(grid.shape[0], num_query_needed, replace=False)
        chosen = grid[idx]
    elif grid.shape[0] > 0:
        short = num_query_needed - grid.shape[0]
        extra = _uniform_frustum_fill(
            short,
            partial_np,
            min_radius,
            max_radius,
            z_min,
            z_max,
            min_sep=v_min * 0.6,
            rng=rng,
        )
        chosen = np.vstack([grid, extra])
    else:
        chosen = _uniform_frustum_fill(
            num_query_needed,
            partial_np,
            min_radius,
            max_radius,
            z_min,
            z_max,
            min_sep=v_min * 0.6,
            rng=rng,
        )

    query_t = torch.from_numpy(chosen.astype(np.float32)).to(device)
    return torch.cat([partial_coord, query_t], dim=0)


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
    ):
        """
        Initialize complete diffusion model.
        
        Args:
            encoder: Sonata encoder
            condition_extractor: Conditional feature extractor
            num_timesteps: Total diffusion steps
            schedule: Noise schedule type
            denoising_steps: Steps for inference
        """
        super().__init__()
        
        self.encoder = encoder
        self.condition_extractor = condition_extractor
        self.denoising_steps = denoising_steps
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule=schedule
        )
        
        # Denoising network
        self.denoiser = DenoisingNetwork(
            in_channels=3,
            condition_dim=condition_extractor.out_dim,
            hidden_dims=[64, 128, 256, 512],
            time_embed_dim=128
        )
    
    def forward(
        self,
        partial_scan: Dict[str, torch.Tensor],
        complete_scan: torch.Tensor,
        complete_batch: torch.Tensor = None,
        return_loss: bool = True,
        condition_features: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            partial_scan: Incomplete input scan
            complete_scan: Ground truth complete scene (N, 3)
            return_loss: Whether to compute loss
            
        Returns:
            Dictionary with predictions and losses
        """
        if condition_features is not None:
            # Use pre-mapped condition features (from precomputed data)
            cond_mapped = condition_features
        else:
            # Extract conditional features and map to complete_scan coords
            cond_features, _ = self.condition_extractor(partial_scan)
            cond_mapped = knn_interpolate(
                cond_features, partial_scan['coord'], complete_scan
            )
        
        # Handle batched data
        if complete_batch is not None:
            batch_size = complete_batch.max().item() + 1
        else:
            batch_size = 1

        t = torch.randint(
            0, self.scheduler.num_timesteps,
            (batch_size,), device=complete_scan.device
        )

        # Per-point timesteps and noise scaling
        noise = torch.randn_like(complete_scan)
        if complete_batch is not None:
            t_per_point = t[complete_batch]
        else:
            t_per_point = t.expand(complete_scan.shape[0])

        dev = complete_scan.device
        sa = self.scheduler.sqrt_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
        som = self.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
        noisy_scan = sa * complete_scan + som * noise

        # Predict residual delta (x0 = x_t + delta) — more stable than noise prediction
        pred_delta = self.denoiser(
            noisy_scan, complete_scan, t_per_point, {'features': cond_mapped}
        )
        target_delta = complete_scan - noisy_scan

        if return_loss:
            pred_x0 = noisy_scan + pred_delta
            loss_delta = F.mse_loss(pred_delta, target_delta)
            loss_x0 = F.mse_loss(pred_x0, complete_scan)
            loss_chamfer = chamfer_distance_training(pred_x0, complete_scan)
            t_norm = t.float().mean() / float(self.scheduler.num_timesteps)
            geom_weight = 1.0 - t_norm
            loss = (
                1.0 * loss_delta
                + 0.1 * loss_x0
                + geom_weight * 0.02 * loss_chamfer
            )
            return {
                'loss': loss,
                'pred_delta': pred_delta,
                'pred_x0': pred_x0,
                'loss_delta': loss_delta,
                'loss_x0': loss_x0,
                'loss_chamfer': loss_chamfer,
            }
        return {'pred_delta': pred_delta}
    
    @torch.no_grad()
    def complete_scene(
        self,
        partial_scan: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        use_query_points: bool = True,
        query_max_radius: float = QUERY_MAX_RADIUS,
        query_min_radius: float = QUERY_MIN_RADIUS,
        query_voxel_size: float = 0.15,
        target_total_points: int = DEFAULT_TARGET_TOTAL_POINTS,
        anchor_alpha: float = 0.9,
    ) -> torch.Tensor:
        """
        Complete scene from partial scan (inference).

        If use_query_points=True, generates query grid in [query_min_radius,
        query_max_radius] (20m by default, aligned with map_from_scans_boost)
        to allow hallucinating occluded regions.

        Args:
            partial_scan: Incomplete input scan
            num_steps: Number of denoising steps (default: self.denoising_steps)
            use_query_points: Add query points in frustum for true completion
            query_max_radius: Max radius (m), default 20
            query_min_radius: Min radius (m), default 3.5
            query_voxel_size: Grid step for query points
            target_total_points: Partial + query count (missing = target - N_partial)
            anchor_alpha: Soft constraint strength for observed points (0–1), default 0.9

        Returns:
            Completed scene point cloud (N, 3)
        """
        if num_steps is None:
            num_steps = self.denoising_steps

        # Extract conditional features (at partial positions)
        cond_features, _ = self.condition_extractor(partial_scan)
        partial_coord = partial_scan["coord"]

        if use_query_points:
            coords = generate_query_points(
                partial_coord,
                max_radius=query_max_radius,
                min_radius=query_min_radius,
                voxel_size=query_voxel_size,
                target_total_points=target_total_points,
            )
            cond_mapped = knn_interpolate(cond_features, partial_coord, coords)
        else:
            coords = partial_coord
            cond_mapped = cond_features

        # Known (observed) points: first num_partial correspond to original partial scan
        num_partial = partial_coord.shape[0]
        known_mask = torch.zeros(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        known_mask[:num_partial] = True

        # Start from coords + small noise (stable init, not pure random)
        noise = torch.randn_like(coords)
        x_t = coords + noise * 0.1

        # Soft constraint: anchor observed points without hard reset
        alpha = anchor_alpha

        # Denoise step by step
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=x_t.device,
        )

        for t in timesteps:
            x_next = self.scheduler.p_sample_step(
                self.denoiser,
                x_t,
                coords,
                t.item(),
                {"features": cond_mapped},
            )
            x_t = x_next.clone()
            # Soft constraint for known (observed) points
            x_t[known_mask] = (
                alpha * partial_coord
                + (1.0 - alpha) * x_next[known_mask]
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
