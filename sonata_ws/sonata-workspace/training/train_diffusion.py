"""
Training Script for Sonata-LiDiff Semantic Scene Completion

Main training loop for the diffusion model with Sonata encoder.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm
from typing import Dict
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import (
    DEFAULT_TRAIN_MAX_POINTS,
    QUERY_MAX_RADIUS,
    QUERY_MIN_RADIUS,
    SceneCompletionDiffusion,
    build_query_shell,
    chamfer_symmetric_torch,
    farthest_point_sample,
    knn_interpolate,
)
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Sonata-LiDiff for semantic scene completion'
    )

    # Data
    parser.add_argument(
        '--data_path', type=str,
        default=os.path.expanduser('~/Simon_ws/dataset/SemanticKITTI/dataset'),
        help='Path to SemanticKITTI dataset root (sequences/, ground_truth/)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=6,
        help='DataLoader worker processes (5–6+ helps feed GPU on A40; 0 = main process only)'
    )
    parser.add_argument(
        '--voxel_size', type=float, default=0.05,
        help='Voxel size for scene representation'
    )
    parser.add_argument(
        '--gt_subdir', type=str, default='ground_truth_v2',
        help='GT maps folder under data root (ground_truth_v2 → local/cropped maps)'
    )
    parser.add_argument(
        '--gt_name_suffix', type=str, default='_v2',
        help='GT filename suffix before .npz (e.g. 000000_v2.npz)'
    )
    parser.add_argument(
        '--scene_radius', type=float, default=20.0,
        help='Crop raw LiDAR + GT map to ego-radius (m); use 0 only if maps/scans are uncropped'
    )
    parser.add_argument(
        '--sequences', type=str, default=None,
        help='Comma-separated seq ids (e.g. 00). One sequence required if using --sequence_scan_* .'
    )
    parser.add_argument(
        '--sequence_scan_start', type=int, default=None,
        help='Inclusive scan index (sorted velodyne filenames in that sequence)'
    )
    parser.add_argument(
        '--sequence_scan_end', type=int, default=None,
        help='Exclusive scan index (same semantics as Python slice end)'
    )
    parser.add_argument(
        '--max_points', type=int, default=20000,
        help='Max points per sample after voxelize (partial and GT)'
    )
    parser.add_argument(
        '--skip_val', action='store_true',
        help='Skip validation loop (no val loss / best_model by val)'
    )

    # Model
    parser.add_argument(
        '--encoder_ckpt', type=str, default='facebook/sonata',
        help='Sonata encoder checkpoint'
    )
    parser.add_argument(
        '--freeze_encoder', action='store_true',
        help='Freeze Sonata encoder weights'
    )
    parser.add_argument(
        '--enable_flash', action='store_true',
        help='Enable flash attention in encoder'
    )
    parser.add_argument(
        '--num_timesteps', type=int, default=1000,
        help='Number of diffusion timesteps'
    )
    parser.add_argument(
        '--schedule', type=str, default='cosine',
        choices=['linear', 'cosine', 'sigmoid'],
        help='Noise schedule type'
    )

    # Training
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--gradient_clip', type=float, default=1.0,
        help='Gradient clipping threshold'
    )
    parser.add_argument(
        '--accumulation_steps', type=int, default=1,
        help='Gradient accumulation steps'
    )

    parser.add_argument(
        '--fp16', action='store_true',
        help='Use mixed precision training (fp16)'
    )
    parser.add_argument(
        '--log-conditioning', action='store_true', dest='log_conditioning',
        help='Print cond_mapped mean/std each training/val forward (verbose).'
    )

    # Output
    parser.add_argument(
        '--output_dir', type=str, default='checkpoints/diffusion',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs/diffusion',
        help='TensorBoard log directory'
    )
    parser.add_argument(
        '--save_freq', type=int, default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--eval_freq', type=int, default=1,
        help='Evaluate every N epochs'
    )

    # Resume
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint'
    )

    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )

    # Precomputed features mode
    parser.add_argument(
        '--precomputed', action='store_true', default=False,
        help='Use precomputed encoder features (skip encoder during training)'
    )
    parser.add_argument(
        '--num_query_extra', type=int, default=DEFAULT_TRAIN_MAX_POINTS,
        help='Random query points per sample (partial voxels + this)'
    )
    parser.add_argument(
        '--query_radius', type=float, default=QUERY_MAX_RADIUS,
        help='Cylinder radius (m) for query shell sampling'
    )
    parser.add_argument(
        '--query_min_radius', type=float, default=QUERY_MIN_RADIUS,
        help='Synthetic queries use r_xy in [min, query_radius] (excludes sensor blind zone)'
    )
    parser.add_argument(
        '--chamfer_max_points', type=int, default=4096,
        help='Max points per cloud when computing Chamfer in loss'
    )
    parser.add_argument(
        '--chamfer_lambda', type=float, default=0.1,
        help='Weight for Chamfer loss on predicted x0'
    )
    parser.add_argument(
        '--center_lambda', type=float, default=0.02,
        help='Weight for centroid alignment loss'
    )
    parser.add_argument(
        '--conditioning-mode', type=str, default='concat',
        choices=['concat', 'additive', 'film'], dest='conditioning_mode',
        help='Denoiser fusion of encoder features (concat=default, additive, film)',
    )
    parser.add_argument(
        '--conditioning-scale', type=float, default=1.0, dest='conditioning_scale',
        help='Strength for additive / FiLM paths',
    )
    parser.add_argument(
        '--condition-dropout-prob', type=float, default=0.1, dest='condition_dropout_prob',
        help='Training-only: P(zero cond), classifier-free style. 0 disables. Unused in eval/inference.',
    )
    parser.add_argument(
        '--conditioning-loss-weight',
        type=float,
        default=0.1,
        dest='conditioning_loss_weight',
        help=(
            'Training-only: same x_t,t,noise; loss_cond=MSE(pred_cond,noise), loss_zero=MSE(pred_zero,noise); '
            'noise_loss = loss_cond + weight * loss_zero. 0 disables second pass.'
        ),
    )

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config
        for key, value in config.items():
            setattr(args, key, value)

    return args


def build_model(args):
    """Build the complete scene completion model."""

    if getattr(args, 'precomputed', False):
        # Precomputed mode: lightweight model without encoder
        from models.diffusion_module import (
            DenoisingNetwork,
            DiffusionScheduler,
            PRED_NOISE_CLAMP,
            TRAIN_TIMESTEP_MIN,
        )
        print('Building model (precomputed features mode)...')

        class PrecomputedModel(torch.nn.Module):
            """Precomputed cond features live on GT points; use FPS proxy partial + query shell."""

            def __init__(
                self,
                denoiser,
                scheduler,
                num_extra,
                radius,
                min_r,
                c_lambda,
                ctr_lambda,
                cmax,
                conditioning_loss_weight: float = 0.0,
            ):
                super().__init__()
                self.denoiser = denoiser
                self.scheduler = scheduler
                self.num_query_extra = num_extra
                self.query_radius = radius
                self.query_min_radius = min_r
                self.chamfer_lambda = c_lambda
                self.center_lambda = ctr_lambda
                self.chamfer_max_points = cmax
                self.conditioning_loss_weight = float(conditioning_loss_weight)

            def forward(self, partial_scan, complete_scan, complete_batch=None,
                        return_loss=True, condition_features=None, **kwargs):
                import torch.nn.functional as F
                clw = float(
                    kwargs.get("conditioning_loss_weight", self.conditioning_loss_weight)
                )
                cond = condition_features
                if complete_batch is not None:
                    batch_size = int(complete_batch.max().item()) + 1
                else:
                    batch_size = 1
                t_hi = self.scheduler.num_timesteps
                t_lo = TRAIN_TIMESTEP_MIN if t_hi > TRAIN_TIMESTEP_MIN else 0
                t = torch.randint(
                    t_lo, t_hi, (batch_size,), device=complete_scan.device,
                )
                total = 0.0
                total_mse = 0.0
                total_ch = 0.0
                total_ctr = 0.0
                total_main = 0.0
                total_lc = 0.0
                total_lz = 0.0
                total_adv = 0.0
                use_aux = self.training and clw > 0.0 and return_loss
                for b in range(batch_size):
                    if complete_batch is not None:
                        mask = complete_batch == b
                        coords_b = complete_scan[mask]
                        cond_b = cond[mask]
                    else:
                        coords_b = complete_scan
                        cond_b = cond
                    n_proxy = min(2048, coords_b.shape[0])
                    n_proxy = max(1, n_proxy)
                    pidx = farthest_point_sample(coords_b, n_proxy)
                    partial_proxy = coords_b[pidx]
                    query_b = build_query_shell(
                        partial_proxy,
                        num_query=self.num_query_extra,
                        radius=self.query_radius,
                        min_radius=self.query_min_radius,
                        device=coords_b.device,
                    )
                    gt_b = knn_interpolate(coords_b, coords_b, query_b)
                    cond_q = knn_interpolate(cond_b, coords_b, query_b)
                    noise_b = torch.randn_like(gt_b)
                    dev = gt_b.device
                    tb = t[b].expand(gt_b.shape[0])
                    sa = self.scheduler.sqrt_alphas_cumprod.to(dev)[tb].unsqueeze(-1)
                    som = self.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[tb].unsqueeze(-1)
                    noisy_b = sa * gt_b + som * noise_b
                    pred_b = self.denoiser(noisy_b, query_b, tb.long(), {'features': cond_q})
                    pred_b = torch.clamp(pred_b, -PRED_NOISE_CLAMP, PRED_NOISE_CLAMP)
                    if use_aux:
                        pred_z = self.denoiser(
                            noisy_b,
                            query_b,
                            tb.long(),
                            {'features': torch.zeros_like(cond_q)},
                        )
                        pred_z = torch.clamp(pred_z, -PRED_NOISE_CLAMP, PRED_NOISE_CLAMP)
                    else:
                        pred_z = None
                    loss_mse_cond = F.mse_loss(pred_b, noise_b)
                    if use_aux and pred_z is not None:
                        loss_zero = F.mse_loss(pred_z, noise_b)
                        noise_term = loss_mse_cond + clw * loss_zero
                        adv = loss_zero - loss_mse_cond
                    else:
                        loss_zero = None
                        adv = None
                        noise_term = loss_mse_cond
                    ab = self.scheduler.alphas_cumprod.to(dev)[tb].unsqueeze(-1).clamp(min=1e-8)
                    som_ab = self.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[tb].unsqueeze(-1)
                    sqrt_ab = self.scheduler.sqrt_alphas_cumprod.to(dev)[tb].unsqueeze(-1)
                    pred_x0 = (noisy_b - som_ab * pred_b) / sqrt_ab
                    Rc = self.query_radius
                    pred_x0_ch = torch.clamp(pred_x0, -Rc, Rc)
                    loss_ch = chamfer_symmetric_torch(
                        pred_x0_ch, gt_b, max_points=self.chamfer_max_points,
                    )
                    pm, gm = pred_x0_ch.mean(0), gt_b.mean(0)
                    loss_ctr = torch.sum((pm - gm) ** 2)
                    recon_main = (
                        loss_mse_cond
                        + self.chamfer_lambda * loss_ch
                        + self.center_lambda * loss_ctr
                    )
                    sample_loss = (
                        noise_term
                        + self.chamfer_lambda * loss_ch
                        + self.center_lambda * loss_ctr
                    ) / batch_size
                    if sample_loss.requires_grad:
                        sample_loss.backward()
                    total += sample_loss.item()
                    total_mse += (loss_mse_cond / batch_size).detach().item()
                    total_ch += (loss_ch / batch_size).detach().item()
                    total_ctr += (loss_ctr / batch_size).detach().item()
                    total_main += (recon_main / batch_size).detach().item()
                    total_lc += (loss_mse_cond / batch_size).detach().item()
                    if use_aux and adv is not None:
                        total_lz += (loss_zero / batch_size).detach().item()
                        total_adv += (adv / batch_size).detach().item()
                if return_loss:
                    return {
                        'loss': total,
                        'main_loss': total_main,
                        'loss_cond': total_lc,
                        'loss_zero': total_lz if use_aux else None,
                        'conditioning_advantage': total_adv if use_aux else None,
                        'loss_mse': total_mse,
                        'loss_chamfer': total_ch,
                        'loss_center': total_ctr,
                        'pred_noise': None,
                    }
                return {'pred_noise': None}

        denoiser = DenoisingNetwork(
            in_channels=3, condition_dim=256,
            hidden_dims=[64, 128, 256, 512],
            time_embed_dim=128, num_neighbors=16,
            conditioning_mode=str(getattr(args, 'conditioning_mode', 'concat')),
            conditioning_scale=float(getattr(args, 'conditioning_scale', 1.0)),
        )
        scheduler = DiffusionScheduler(args.num_timesteps, args.schedule)
        model = PrecomputedModel(
            denoiser, scheduler,
            num_extra=getattr(args, 'num_query_extra', DEFAULT_TRAIN_MAX_POINTS),
            radius=float(getattr(args, 'query_radius', QUERY_MAX_RADIUS)),
            min_r=float(getattr(args, 'query_min_radius', QUERY_MIN_RADIUS)),
            c_lambda=float(getattr(args, 'chamfer_lambda', 0.1)),
            ctr_lambda=float(getattr(args, 'center_lambda', 0.02)),
            cmax=int(getattr(args, 'chamfer_max_points', 4096)),
            conditioning_loss_weight=float(
                getattr(args, 'conditioning_loss_weight', 0.0)
            ),
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        return model

    # Full model with Sonata encoder
    print("Loading Sonata encoder...")
    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=args.freeze_encoder,
        enable_flash=args.enable_flash,
        feature_levels=[0]
    )

    # Conditional feature extractor
    print("Building conditional feature extractor...")
    condition_extractor = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[0],
        fusion_type="concat"
    )

    # Complete diffusion model
    print("Building diffusion model...")
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        denoising_steps=50,
        chamfer_max_points=args.chamfer_max_points,
        num_query_extra=args.num_query_extra,
        query_radius=args.query_radius,
        query_min_radius=args.query_min_radius,
        chamfer_lambda=args.chamfer_lambda,
        center_lambda=args.center_lambda,
        conditioning_mode=str(getattr(args, 'conditioning_mode', 'concat')),
        conditioning_scale=float(getattr(args, 'conditioning_scale', 1.0)),
        condition_dropout_prob=float(getattr(args, 'condition_dropout_prob', 0.1)),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def train_epoch(
    model: SceneCompletionDiffusion,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    args,
    writer: SummaryWriter,
    scaler: torch.cuda.amp.GradScaler = None,
) -> float:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for i, batch in enumerate(pbar):
        dev = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(dev, non_blocking=True)

        # Prepare input
        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }

        complete_coord = batch['complete_coord']
        complete_batch = batch.get('complete_batch')
        if complete_batch is not None:
            complete_batch = complete_batch.to(dev, non_blocking=True)

        cond_feat = batch.get('condition_features', None)
        if cond_feat is not None:
            cond_feat = cond_feat.to(dev, non_blocking=True)
            if cond_feat.dtype == torch.float16:
                cond_feat = cond_feat.float()

        # Forward pass (with optional mixed precision)
        use_amp = getattr(args, 'fp16', False) and dev.type == 'cuda'
        clw = float(getattr(args, 'conditioning_loss_weight', 0.0))
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(
                partial_scan, complete_coord, complete_batch,
                return_loss=True, condition_features=cond_feat,
                log_conditioning=getattr(args, 'log_conditioning', False),
                conditioning_loss_weight=clw,
            )
            loss = output['loss']

        # Backward (with optional mixed precision)
        if isinstance(loss, torch.Tensor):
            loss = loss / args.accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_val = loss.item() * args.accumulation_steps
        else:
            loss_val = loss

        # Update weights
        if (i + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip
                    )
                optimizer.step()
            optimizer.zero_grad()

        # Logging
        total_loss += loss_val
        postfix = {'loss': loss_val}
        if isinstance(output, dict):
            ml = output.get('main_loss')
            if ml is not None:
                postfix['main_loss'] = float(ml) if isinstance(ml, torch.Tensor) else float(ml)
            lc = output.get('loss_cond')
            if lc is not None:
                postfix['loss_cond'] = float(lc) if isinstance(lc, torch.Tensor) else float(lc)
            lz = output.get('loss_zero')
            if lz is not None:
                postfix['loss_zero'] = float(lz) if isinstance(lz, torch.Tensor) else float(lz)
            ca = output.get('conditioning_advantage')
            if ca is not None:
                postfix['cond_adv'] = float(ca) if isinstance(ca, torch.Tensor) else float(ca)
            if output.get('loss_chamfer') is not None:
                ch = output['loss_chamfer']
                postfix['chamfer'] = float(ch) if isinstance(ch, torch.Tensor) else ch
            if output.get('pred_x0_mean') is not None:
                postfix['px0_m'] = float(output['pred_x0_mean'])
        pbar.set_postfix(postfix)

        # TensorBoard logging
        step = epoch * len(dataloader) + i
        writer.add_scalar('train/loss', loss_val, step)
        if isinstance(output, dict):
            ml = output.get('main_loss')
            if ml is not None:
                writer.add_scalar(
                    'train/main_loss',
                    float(ml) if isinstance(ml, torch.Tensor) else float(ml),
                    step,
                )
            lc = output.get('loss_cond')
            if lc is not None:
                writer.add_scalar(
                    'train/loss_cond',
                    float(lc) if isinstance(lc, torch.Tensor) else float(lc),
                    step,
                )
            lz = output.get('loss_zero')
            if lz is not None:
                writer.add_scalar(
                    'train/loss_zero',
                    float(lz) if isinstance(lz, torch.Tensor) else float(lz),
                    step,
                )
            ca = output.get('conditioning_advantage')
            if ca is not None:
                writer.add_scalar(
                    'train/conditioning_advantage',
                    float(ca) if isinstance(ca, torch.Tensor) else float(ca),
                    step,
                )
            if 'loss_mse' in output:
                writer.add_scalar('train/loss_mse', float(output['loss_mse']), step)
            if 'loss_chamfer' in output:
                writer.add_scalar('train/chamfer', float(output['loss_chamfer']), step)
            if 'loss_center' in output:
                writer.add_scalar('train/loss_center', float(output['loss_center']), step)
            if 'pred_x0_mean' in output:
                writer.add_scalar('train/pred_x0_mean', float(output['pred_x0_mean']), step)
            if 'pred_x0_std' in output:
                writer.add_scalar('train/pred_x0_std', float(output['pred_x0_std']), step)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(
    model: SceneCompletionDiffusion,
    dataloader: DataLoader,
    epoch: int,
    args,
    writer: SummaryWriter
) -> float:
    """Validate the model."""

    model.eval()
    total_loss = 0.0
    output = {}

    pbar = tqdm(dataloader, desc=f"Validation")

    for i, batch in enumerate(pbar):
        dev = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(dev, non_blocking=True)

        # Prepare input
        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }

        complete_coord = batch['complete_coord']
        complete_batch = batch.get('complete_batch')
        if complete_batch is not None:
            complete_batch = complete_batch.to(dev, non_blocking=True)

        cond_feat = batch.get('condition_features', None)
        if cond_feat is not None:
            cond_feat = cond_feat.to(dev, non_blocking=True)
            if cond_feat.dtype == torch.float16:
                cond_feat = cond_feat.float()

        use_amp = getattr(args, 'fp16', False) and dev.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(
                partial_scan, complete_coord, complete_batch,
                return_loss=True, condition_features=cond_feat,
                log_conditioning=getattr(args, 'log_conditioning', False),
                conditioning_loss_weight=0.0,
            )
            loss = output['loss']

        lv = loss if isinstance(loss, float) else loss.item()
        total_loss += lv
        pf = {'loss': lv}
        if isinstance(output, dict) and output.get('loss_chamfer') is not None:
            ch = output['loss_chamfer']
            pf['chamfer'] = float(ch) if isinstance(ch, torch.Tensor) else ch
        pbar.set_postfix(pf)

    avg_loss = total_loss / len(dataloader)

    # TensorBoard logging
    writer.add_scalar('val/loss', avg_loss, epoch)
    # Last batch aux metrics (cheap sanity check)
    if isinstance(output, dict):
        if output.get('loss_chamfer') is not None:
            ch = output['loss_chamfer']
            writer.add_scalar('val/chamfer', float(ch) if isinstance(ch, torch.Tensor) else ch, epoch)

    return avg_loss


def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Arguments: {args}")

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Build datasets
    print("\nLoading datasets...")
    use_precomputed = getattr(args, 'precomputed', False)

    def _common_ds_kwargs():
        kw = dict(
            root=args.data_path,
            voxel_size=args.voxel_size,
            max_points=args.max_points,
            use_ground_truth_maps=True,
            use_precomputed=use_precomputed,
            gt_subdir=args.gt_subdir,
            gt_name_suffix=args.gt_name_suffix,
            scene_radius=args.scene_radius,
        )
        if args.sequences:
            kw['sequence_ids'] = [
                s.strip() for s in args.sequences.split(',') if s.strip()
            ]
        return kw

    train_kw = _common_ds_kwargs()
    if args.sequence_scan_start is not None:
        train_kw['sequence_scan_start'] = args.sequence_scan_start
    if args.sequence_scan_end is not None:
        train_kw['sequence_scan_end'] = args.sequence_scan_end

    train_dataset = SemanticKITTI(
        split='train',
        augmentation=not use_precomputed,
        **train_kw,
    )

    val_loader = None
    if not args.skip_val:
        val_kw = _common_ds_kwargs()
        if (
            args.sequence_scan_start is not None
            and args.sequence_scan_end is not None
            and args.sequences
        ):
            # Hold out the next scans after the train window (same sequence).
            val_kw['sequence_scan_start'] = args.sequence_scan_end
            val_kw['sequence_scan_end'] = args.sequence_scan_end + 20
        val_dataset = SemanticKITTI(
            split='train' if args.sequences else 'val',
            augmentation=False,
            **val_kw,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    model = build_model(args).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume is not None:
        print(f"\nResuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(
        enabled=getattr(args, 'fp16', False) and device.type == 'cuda'
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, args, writer, scaler
        )
        logger.info(f"Train loss: {train_loss:.6f}")

        # Validate
        if val_loader is not None and (epoch + 1) % args.eval_freq == 0:
            val_loss = validate(
                model, val_loader, epoch, args, writer
            )
            logger.info(f"Val loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    args.output_dir, 'best_model.pth'
                )
                save_checkpoint(
                    save_path, model, optimizer, scheduler,
                    epoch, best_val_loss
                )
                logger.info(f"Saved best model to {save_path}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                save_path, model, optimizer, scheduler,
                epoch, best_val_loss
            )
            logger.info(f"Saved checkpoint to {save_path}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

    # Save final model
    save_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(
        save_path, model, optimizer, scheduler,
        args.num_epochs - 1, best_val_loss
    )
    logger.info(f"\nTraining completed! Final model saved to {save_path}")

    writer.close()


if __name__ == "__main__":
    main()