"""
VAE-GAN training: PointCloudVAE V3 + WGAN-GP discriminator.

The VAE V3 (multi-token, 32 latent tokens) is kept intact and acts as the
generator.  A multi-scale PointNet critic is trained adversarially with
Wasserstein loss + gradient penalty (WGAN-GP) + feature-matching loss.

Generator (VAE) loss per step:
    L_G = L_chamfer + beta_kl * L_KL
        + lambda_adv * L_wasserstein_gen
        + lambda_fm  * L_feature_match
    L_wasserstein_gen = -E[D(fake)]   (fool the critic)
    L_feature_match   = sum_l L1(D_l(fake), D_l(real).detach())

Critic (discriminator) loss, n_critic steps per generator step:
    L_D = E[D(fake)] - E[D(real)] + lambda_gp * GP
    GP  = E[(||∇D(x̂)||₂ - 1)²]  on interpolated samples

Publication-quality additions (defaults ON):
  • Feature-matching loss (Salimans 2016, Larsen 2016): stabilises training
    and improves sample quality beyond pure adversarial loss.
  • EMA of VAE weights (StyleGAN/BigGAN convention): a running average of
    the generator is maintained throughout training and saved alongside
    the live weights.  Use EMA weights at inference time for ~0.5–2 dB
    better reconstruction quality.

Training schedule:
    Epochs 0 … disc_warmup_epochs-1                          : VAE-only (no adv loss)
    Epochs disc_warmup_epochs … disc_warmup+ramp_epochs-1    : adv loss linearly
                                                                ramped 0 → lambda_adv
    Epochs disc_warmup+ramp_epochs … end                     : full VAE-GAN

The disc-warmup ensures the VAE has learned a reasonable reconstruction
before the discriminator starts pushing it.  The adversarial ramp avoids
the step discontinuity at epoch=disc_warmup_epochs that otherwise destabilises
fine-tunes of an already-converged VAE V3 (the critic dominates and val
regresses).  Set --lambda_adv_ramp_epochs 0 to disable the ramp (legacy
step-function behaviour).

Recommended usage:
    # Fine-tune from a pre-trained VAE V3 checkpoint (strongly recommended):
    python training/train_vae_gan.py --resume_vae checkpoints/point_vae_v3/best_point_vae.pth

    # Train from scratch (longer convergence):
    python training/train_vae_gan.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

from data.semantickitti import SemanticKITTI, collate_fn
from models.discriminator import (
    MultiScalePointDiscriminator,
    feature_matching_loss,
    gradient_penalty,
)
from models.critic_token import (
    TokenCritic,
    feature_matching_loss_tokens,
    r1_penalty,
)
from models.point_cloud_vae import PointCloudVAE, kl_divergence
from models.refinement_net import chamfer_distance
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# EMA of generator weights (StyleGAN-style)
# ---------------------------------------------------------------------------

class WeightEMA:
    """
    Exponential moving average of a model's parameters.

    Maintains a separate "shadow" copy of the generator that decays toward the
    live weights at each step:
        ema_w  ←  decay * ema_w  +  (1 - decay) * live_w

    Inference / evaluation should use these EMA weights — they consistently
    yield lower reconstruction error than the live weights in modern GANs
    (Karras et al. StyleGAN2, Brock et al. BigGAN).

    The shadow model is in `eval()` mode and never receives gradients.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. None here, but safe-guard) — copy from live.
        for b_ema, b in zip(self.shadow.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train VAE-GAN: PointCloudVAE V3 + WGAN-GP discriminator"
    )
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)

    # Generator (VAE) optimiser — matches V3 baseline
    p.add_argument("--lr_gen", type=float, default=3e-4)
    p.add_argument("--weight_decay_gen", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="LR warmup for the generator scheduler")

    # Critic (discriminator) optimiser — Adam(beta1=0) as per WGAN-GP paper
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=5,
                   help="Discriminator update steps per generator step")
    p.add_argument("--lambda_gp", type=float, default=10.0,
                   help="Gradient penalty coefficient (WGAN-GP, default 10)")
    p.add_argument("--critic_drift", type=float, default=0.0,
                   help="Weight of the WGAN drift penalty eps*E[D(real)^2] "
                        "(Karras et al. 2018). Anchors the critic score scale. "
                        "R1 constrains gradients at real samples but not the "
                        "score magnitude, so critics whose scores run away "
                        "need this to stay bounded.")
    p.add_argument("--lambda_adv", type=float, default=0.01,
                   help="Adversarial loss weight (target) in generator loss. "
                        "Lowered from 0.1 → 0.01 for safer fine-tunes of a converged V3.")
    p.add_argument("--lambda_adv_ramp_epochs", type=int, default=5,
                   help="Linearly ramp lambda_adv from 0 to its target over this many epochs, "
                        "starting at epoch=disc_warmup_epochs. 0 disables the ramp.")
    p.add_argument("--lambda_fm", type=float, default=10.0,
                   help="Feature-matching loss weight (Salimans 2016). Set to 0 to disable.")
    p.add_argument("--disc_warmup_epochs", type=int, default=10,
                   help="Epochs of VAE-only training before enabling the discriminator")

    # --- Critic redesign (root-cause fixes; defaults preserve legacy behaviour) ---
    p.add_argument("--critic", choices=["multiscale", "token"], default="multiscale",
                   help="multiscale = legacy global-maxpool critic; "
                        "token = localized cross-attention K-token critic (mirrors V3).")
    p.add_argument("--critic_tokens", type=int, default=32)
    p.add_argument("--critic_dim", type=int, default=256)
    p.add_argument("--critic_layers", type=int, default=2)
    p.add_argument("--conditional_critic", action="store_true", default=False,
                   help="Token critic also cross-attends to the partial scan (judges "
                        "plausibility conditioned on the input scene).")
    p.add_argument("--adaptive_adv", action="store_true", default=False,
                   help="VQGAN-style adaptive adversarial weight: lambda_eff = "
                        "ramp * ||grad_recon|| / (||grad_adv|| + eps) on the last decoder "
                        "layer, so the critic never overwhelms reconstruction.")
    p.add_argument("--adaptive_max", type=float, default=1e4,
                   help="Clamp for the adaptive adversarial weight.")
    p.add_argument("--gp_mode", choices=["interp", "r1"], default="interp",
                   help="interp = legacy WGAN-GP index interpolation (meaningless for "
                        "unordered sets); r1 = R1 penalty on real samples only.")
    p.add_argument("--log_grad_align", action="store_true", default=False,
                   help="Log cosine(grad_recon, grad_adv) on the last decoder layer.")

    # EMA of generator weights (publication-grade GAN practice)
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Maintain EMA of VAE weights (recommended for publication)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false",
                   help="Disable EMA tracking of VAE weights")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay (closer to 1.0 = slower-moving average)")
    p.add_argument("--ema_start_epoch", type=int, default=0,
                   help="Begin EMA tracking only after this epoch (warmup-friendly)")

    # VAE architecture — keep identical to V3
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)

    # VAE loss
    p.add_argument("--beta_kl", type=float, default=1e-3)

    # Validation / checkpoint selection
    p.add_argument("--val_stochastic", action="store_true", default=False,
                   help="Sample z during validation (legacy). Default decodes the posterior "
                        "mean μ deterministically, which is the correct, noise-free signal for "
                        "best-checkpoint selection.")

    # Data
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    # Per-sample normalisation (must match V3 training)
    p.add_argument("--center_gt", action="store_true", default=True)
    p.add_argument("--scale_gt", action="store_true", default=True)

    # IO
    p.add_argument("--output_dir", type=str, default="checkpoints/vae_gan")
    p.add_argument("--log_dir", type=str, default="logs/vae_gan")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--resume_vae", type=str, default=None,
                   help="Pre-trained VAE V3 checkpoint (warm-start, strongly recommended)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume full VAE-GAN from a previous VAE-GAN checkpoint")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="Training device: 'cuda', 'cuda:1', 'cpu'. "
                        "Defaults to cuda if available, otherwise cpu.")

    # GT data version
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--gt_name_suffix", type=str, default="")

    # Reproducibility
    p.add_argument("--seed", type=int, default=None,
                   help="Set torch/numpy/random seeds + DataLoader worker seeding.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_points(
    pts: torch.Tensor, center: bool = True, scale: bool = True
):
    """Center and optionally scale a point cloud to [-1, 1]. Returns (pts_norm, centroid, scale)."""
    centroid = pts.mean(dim=0) if center else torch.zeros(3, device=pts.device)
    pts_c = pts - centroid
    s = pts_c.abs().max().clamp(min=1e-6) if scale else torch.ones(1, device=pts.device)
    return pts_c / s, centroid, s


def lambda_adv_effective(epoch: int, args) -> float:
    """
    Effective lambda_adv at a given epoch, applying the linear ramp from 0 to
    args.lambda_adv over args.lambda_adv_ramp_epochs starting at
    args.disc_warmup_epochs.

    At epoch == disc_warmup_epochs the ramp emits 1/ramp_epochs of the target
    (i.e. small but non-zero), reaching the full target at epoch
    (disc_warmup_epochs + ramp_epochs - 1).  Set ramp_epochs <= 0 to disable
    the ramp (legacy step-function behaviour).
    """
    if epoch < args.disc_warmup_epochs:
        return 0.0
    if args.lambda_adv_ramp_epochs <= 0:
        return args.lambda_adv
    progress = (epoch - args.disc_warmup_epochs + 1) / args.lambda_adv_ramp_epochs
    return args.lambda_adv * min(1.0, max(0.0, progress))


def sample_to_n(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Random subsample or tile to exactly n points."""
    N = pts.size(0)
    if N == n:
        return pts
    if N > n:
        return pts[torch.randperm(N, device=pts.device)[:n]]
    # tile if too few points
    repeats = (n + N - 1) // N
    return pts.repeat(repeats, 1)[:n]


def build_norm_batch(cc, cb, bsz, center, scale, n_pts, device):
    """
    Builds per-sample normalised point cloud list and a stacked (B, n_pts, 3)
    real batch subsampled to n_pts for the discriminator. Also returns the
    per-sample (centroid, scale) so conditioning clouds can be put in the same
    normalised frame.
    """
    pts_list, norms = [], []
    for b in range(bsz):
        pts = cc[cb == b]
        pts_n, c, s = normalize_points(pts, center, scale)
        pts_list.append(pts_n)
        norms.append((c, s))
    real_batch = torch.stack([sample_to_n(p, n_pts) for p in pts_list])
    return pts_list, real_batch, norms


def build_ctx_batch(pc, pbatch, norms, bsz, n_pts, device):
    """Normalise each sample's partial scan with the complete cloud's
    (centroid, scale) and stack to (B, n_pts, 3) for a conditional critic."""
    ctx_list = []
    for b in range(bsz):
        pts = pc[pbatch == b]
        c, s = norms[b]
        if pts.size(0) == 0:
            pts = torch.zeros(1, 3, device=device)
        ctx_list.append(sample_to_n((pts - c) / s, n_pts))
    return torch.stack(ctx_list)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    vae, disc, opt_gen, opt_disc, loader, epoch, args, writer, device,
    ema: "WeightEMA | None" = None,
):
    vae.train()
    disc.train()

    use_adv = epoch >= args.disc_warmup_epochs
    use_fm = use_adv and args.lambda_fm > 0
    lambda_adv_eff = lambda_adv_effective(epoch, args)
    K = args.num_decoded_points

    stats = dict(gen=0.0, disc=0.0, chamfer=0.0, kl=0.0, adv=0.0, fm=0.0,
                 adapt_w=0.0, cos=0.0)
    cos_n = 0
    n_batches = 0

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        pts_list, real_batch, norms = build_norm_batch(
            cc, cb, bsz, args.center_gt, args.scale_gt, K, device
        )
        ctx_batch = None
        if args.conditional_critic:
            ctx_batch = build_ctx_batch(
                batch["partial_coord"], batch["partial_batch"], norms, bsz, K, device
            )

        def disc_fwd(x, **kw):
            return disc(x, ctx_batch, **kw) if ctx_batch is not None else disc(x, **kw)

        # ----------------------------------------------------------------
        # Discriminator steps (n_critic per generator step)
        # ----------------------------------------------------------------
        if use_adv:
            disc.requires_grad_(True)
            vae.requires_grad_(False)

            disc_loss_accum = 0.0
            for _ in range(args.n_critic):
                with torch.no_grad():
                    fake_list = []
                    for pts_n in pts_list:
                        mu, logvar = vae.encode(pts_n)
                        z = vae.reparameterize(mu, logvar)
                        fake_list.append(vae.decode(z))
                    fake_batch = torch.stack(fake_list)   # (B, K, 3)

                opt_disc.zero_grad()
                real_score = disc_fwd(real_batch)             # (B, 1)
                fake_score = disc_fwd(fake_batch.detach())    # (B, 1)
                if args.gp_mode == "r1":
                    gp = r1_penalty(disc, real_batch, ctx=ctx_batch)
                else:
                    gp = gradient_penalty(disc, real_batch, fake_batch.detach(), device)
                d_loss = (
                    fake_score.mean() - real_score.mean()
                    + args.lambda_gp * gp
                )
                if args.critic_drift > 0:
                    d_loss = d_loss + args.critic_drift * (real_score ** 2).mean()
                d_loss.backward()
                opt_disc.step()
                disc_loss_accum += d_loss.item()

            stats["disc"] += disc_loss_accum / args.n_critic
            disc.requires_grad_(False)
            vae.requires_grad_(True)

        # ----------------------------------------------------------------
        # Generator (VAE) step
        # ----------------------------------------------------------------
        opt_gen.zero_grad()

        recon_list, mu_list, logvar_list = [], [], []
        for pts_n in pts_list:
            mu, logvar = vae.encode(pts_n)
            z = vae.reparameterize(mu, logvar)
            recon_list.append(vae.decode(z))
            mu_list.append(mu)
            logvar_list.append(logvar)

        # Chamfer: per-sample mean
        chamfer_losses = [
            chamfer_distance(recon_list[b], pts_list[b]) for b in range(bsz)
        ]
        chamfer_term = torch.stack(chamfer_losses).mean()

        # KL divergence across batch
        mu_all = torch.stack(mu_list)           # (B, latent_dim)
        logvar_all = torch.stack(logvar_list)   # (B, latent_dim)
        kl_term = kl_divergence(mu_all, logvar_all)

        g_loss = chamfer_term + args.beta_kl * kl_term

        fm_value = 0.0
        adapt_w_val = 1.0
        cos_val = None
        if use_adv:
            fake_batch_g = torch.stack(recon_list)          # (B, K, 3) — with grads
            if use_fm:
                # Single D forward pass on each side, taking score + features
                fake_score_g, fake_feats = disc_fwd(fake_batch_g, return_features=True)
                with torch.no_grad():
                    _, real_feats = disc_fwd(real_batch, return_features=True)
                adv_loss = -fake_score_g.mean()
                fm_loss = feature_matching_loss(real_feats, fake_feats)
            else:
                adv_loss = -disc_fwd(fake_batch_g).mean()   # Wasserstein gen loss
                fm_loss = None

            # VQGAN-style adaptive adversarial weight: balance the adversarial
            # gradient against the reconstruction gradient on the last decoder
            # layer, so the critic can never overwhelm reconstruction.
            if args.adaptive_adv:
                last = vae.output_head.weight
                g_rec = torch.autograd.grad(chamfer_term, last, retain_graph=True)[0]
                g_adv = torch.autograd.grad(adv_loss, last, retain_graph=True)[0]
                adapt_w = (g_rec.norm() / (g_adv.norm() + 1e-8)).clamp(0, args.adaptive_max).detach()
                adapt_w_val = adapt_w.item()
                if args.log_grad_align:
                    cos_val = torch.nn.functional.cosine_similarity(
                        g_rec.flatten(), g_adv.flatten(), dim=0).item()
            else:
                adapt_w = 1.0

            w_adv = lambda_adv_eff * adapt_w
            g_loss = g_loss + w_adv * adv_loss
            if fm_loss is not None:
                g_loss = g_loss + args.lambda_fm * fm_loss
                fm_value = fm_loss.item()
                stats["fm"] += fm_value
            stats["adv"] += adv_loss.item()
            stats["adapt_w"] += adapt_w_val
            if cos_val is not None:
                stats["cos"] += cos_val
                cos_n += 1

        g_loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.gradient_clip)
        opt_gen.step()

        # EMA update — track every generator step once enabled
        if ema is not None and epoch >= args.ema_start_epoch:
            ema.update(vae)

        # Ensure disc params are trainable again for next batch
        disc.requires_grad_(True)

        stats["gen"] += g_loss.item()
        stats["chamfer"] += chamfer_term.item()
        stats["kl"] += kl_term.item()
        n_batches += 1

        step = epoch * len(loader) + i
        writer.add_scalar("train/gen_loss", g_loss.item(), step)
        writer.add_scalar("train/chamfer", chamfer_term.item(), step)
        writer.add_scalar("train/kl", kl_term.item(), step)
        if use_adv:
            writer.add_scalar("train/disc_loss", disc_loss_accum / args.n_critic, step)
            writer.add_scalar("train/adv_loss", adv_loss.item(), step)
            writer.add_scalar("train/lambda_adv_eff", lambda_adv_eff, step)
            if args.adaptive_adv:
                writer.add_scalar("train/adaptive_w", adapt_w_val, step)
                writer.add_scalar("train/w_adv_eff", lambda_adv_eff * adapt_w_val, step)
            if cos_val is not None:
                writer.add_scalar("train/cos_recon_adv", cos_val, step)
            if use_fm:
                writer.add_scalar("train/fm_loss", fm_value, step)

    nb = max(n_batches, 1)
    out = {k: v / nb for k, v in stats.items()}
    out["cos"] = stats["cos"] / max(cos_n, 1) if cos_n else None
    out["lambda_adv_eff"] = lambda_adv_eff
    return out


@torch.no_grad()
def validate(vae, loader, args, device, deterministic: bool = True):
    """
    Mean validation loss (CD + beta_kl·KL) over the val split.

    deterministic=True decodes the posterior mean μ instead of a sampled z.
    This is the correct setting for *best-checkpoint selection*: the metric we
    care about is reconstruction quality, and sampling z injects reparam noise
    that turns the selection signal into a noisy estimate (and, once the
    adversarial phase perturbs logvar, biases it). Pass --val_stochastic to
    restore the legacy sampled behaviour.
    """
    vae.eval()
    tot = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        for b in range(bsz):
            pts = cc[cb == b]
            pts_n, _, _ = normalize_points(pts, args.center_gt, args.scale_gt)
            mu, logvar = vae.encode(pts_n)
            z = mu if deterministic else vae.reparameterize(mu, logvar)
            recon = vae.decode(z)
            cd = chamfer_distance(recon, pts_n)
            kl = kl_divergence(mu.unsqueeze(0), logvar.unsqueeze(0))
            tot += (cd + args.beta_kl * kl).item() / bsz
        n += 1
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_vaegan_checkpoint(
    path, vae, disc, opt_gen, opt_disc, epoch, best_val,
    ema: "WeightEMA | None" = None,
):
    """Save full VAE-GAN state, including EMA shadow weights when enabled."""
    payload = {
        "epoch": epoch,
        "vae_state_dict": vae.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "best_val_loss": best_val,
    }
    if ema is not None:
        # EMA weights — preferred for downstream inference / evaluation
        payload["vae_ema_state_dict"] = ema.state_dict()
    torch.save(payload, path)


def load_vaegan_checkpoint(
    path, vae, disc, opt_gen, opt_disc, device,
    ema: "WeightEMA | None" = None,
):
    ck = torch.load(path, map_location=device)
    vae.load_state_dict(ck["vae_state_dict"])
    disc.load_state_dict(ck["disc_state_dict"])
    opt_gen.load_state_dict(ck["opt_gen_state_dict"])
    opt_disc.load_state_dict(ck["opt_disc_state_dict"])
    if ema is not None and "vae_ema_state_dict" in ck:
        ema.load_state_dict(ck["vae_ema_state_dict"])
    return ck.get("epoch", 0), ck.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Seeded with {args.seed}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ds_kwargs = {}
    if args.gt_subdir != "ground_truth":
        ds_kwargs["gt_subdir"] = args.gt_subdir
    if args.gt_name_suffix:
        ds_kwargs["gt_name_suffix"] = args.gt_name_suffix

    ds_train = SemanticKITTI(
        root=args.data_path, split="train", use_ground_truth_maps=True,
        augmentation=True, use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
        **ds_kwargs,
    )
    ds_val = SemanticKITTI(
        root=args.data_path, split="val", use_ground_truth_maps=True,
        augmentation=False, use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
        **ds_kwargs,
    )

    seed_base = args.seed if args.seed is not None else 0
    def _worker_init(wid: int):
        np.random.seed(seed_base + wid)
        random.seed(seed_base + wid)

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        worker_init_fn=_worker_init,
    )

    # --- Models ---
    vae = PointCloudVAE(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        num_latent_tokens=args.num_latent_tokens,
        internal_dim=args.internal_dim,
        num_heads=args.num_heads,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)

    if args.critic == "token":
        disc = TokenCritic(
            in_dim=3, dim=args.critic_dim, num_tokens=args.critic_tokens,
            num_heads=args.num_heads, num_layers=args.critic_layers,
            conditional=args.conditional_critic,
        ).to(device)
    else:
        disc = MultiScalePointDiscriminator(in_dim=3).to(device)
    logger.info(
        f"Critic: {args.critic}  (adaptive_adv={args.adaptive_adv}, "
        f"gp_mode={args.gp_mode}, conditional={args.conditional_critic})"
    )

    logger.info(f"VAE parameters:           {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters() if p.requires_grad):,}")

    # --- Optimisers ---
    # Generator: AdamW matches V3 baseline
    opt_gen = optim.AdamW(vae.parameters(), lr=args.lr_gen, weight_decay=args.weight_decay_gen)
    # Discriminator: Adam(beta1=0, beta2=0.9) as per WGAN-GP paper
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.0, 0.9))

    # Generator LR schedule: linear warmup → cosine anneal
    warmup_sch = optim.lr_scheduler.LinearLR(
        opt_gen, start_factor=0.01, total_iters=args.warmup_epochs,
    )
    cosine_sch = optim.lr_scheduler.CosineAnnealingLR(
        opt_gen, T_max=max(args.num_epochs - args.warmup_epochs, 1), eta_min=1e-6,
    )
    sch_gen = optim.lr_scheduler.SequentialLR(
        opt_gen, [warmup_sch, cosine_sch], milestones=[args.warmup_epochs],
    )

    start = 0
    best = float("inf")

    # Warm-start generator from pre-trained VAE V3 (recommended)
    if args.resume_vae:
        ck = torch.load(args.resume_vae, map_location=device)
        state = ck.get("model_state_dict", ck)
        vae.load_state_dict(state, strict=True)
        logger.info(f"Loaded pre-trained VAE V3 from {args.resume_vae}")

    # Build EMA shadow AFTER any warm-start so it begins from the right state.
    ema = WeightEMA(vae, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        logger.info(f"EMA tracking enabled (decay={args.ema_decay}, start_epoch={args.ema_start_epoch})")

    # Or resume a previous VAE-GAN run
    if args.resume:
        start, best = load_vaegan_checkpoint(
            args.resume, vae, disc, opt_gen, opt_disc, device, ema=ema,
        )
        start += 1
        logger.info(f"Resumed VAE-GAN from epoch {start - 1}, best_val={best:.6f}")

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start, args.num_epochs):
        s = train_epoch(
            vae, disc, opt_gen, opt_disc, train_loader, epoch, args, writer, device,
            ema=ema,
        )

        # Validate live and (if available) EMA weights — log both.
        deterministic_val = not args.val_stochastic
        va_live = validate(vae, val_loader, args, device, deterministic=deterministic_val)
        va_ema = None
        if ema is not None and epoch >= args.ema_start_epoch:
            va_ema = validate(ema.shadow, val_loader, args, device, deterministic=deterministic_val)
            writer.add_scalar("val/loss_ema", va_ema, epoch)
        writer.add_scalar("val/loss", va_live, epoch)
        writer.add_scalar("lr/gen", opt_gen.param_groups[0]["lr"], epoch)
        sch_gen.step()

        # Use EMA val loss for best-checkpoint selection when available
        # (EMA generally outperforms live weights once warmup is over).
        va = va_ema if va_ema is not None else va_live

        use_adv = epoch >= args.disc_warmup_epochs
        ema_str = f"  val_ema={va_ema:.6f}" if va_ema is not None else ""
        fm_str = f"  fm={s['fm']:.6f}" if s["fm"] > 0 else ""
        cos_str = f"  cos(rec,adv)={s['cos']:+.3f}" if s.get("cos") is not None else ""
        aw_str = f"  adapt_w={s['adapt_w']:.4f}" if s.get("adapt_w", 0) else ""
        adv_str = (
            f"  adv={s['adv']:.6f}  λ_adv={s['lambda_adv_eff']:.4f}{aw_str}{cos_str}"
            if use_adv else ""
        )
        logger.info(
            f"Epoch {epoch:3d} | "
            f"gen={s['gen']:.6f}  disc={s['disc']:.6f}  "
            f"cd={s['chamfer']:.6f}  kl={s['kl']:.4f}"
            f"{adv_str}{fm_str}  "
            f"val={va_live:.6f}{ema_str}  "
            f"adv_mode={'ON' if use_adv else 'OFF'}"
        )

        if va < best:
            best = va
            path = os.path.join(args.output_dir, "best_vae_gan.pth")
            save_vaegan_checkpoint(
                path, vae, disc, opt_gen, opt_disc, epoch, best, ema=ema,
            )
            logger.info(f"  → saved best checkpoint (val={best:.6f})")

        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_vaegan_checkpoint(
                path, vae, disc, opt_gen, opt_disc, epoch, best, ema=ema,
            )

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
