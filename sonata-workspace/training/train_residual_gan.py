"""
Stage-2 CD-leashed residual refinement for the VAE-GAN.

Freeze a CD-optimal Stage-1 VAE (e.g. the recon_only control). Learn ONLY a
small bounded per-point offset (ResidualRefiner) driven by the aligned token
critic, so the adversarial signal can add the JSD/coverage structure it is good
at WITHOUT moving points off the surface (CD is leashed by the bounded offset +
an ||offset||^2 penalty). This is the Stage-2 analogue of LiDiff's refinement.

    refined = frozen_decode(z) + refiner(base, frozen_latent_tokens)
    L_refiner = Chamfer(refined, GT) + gamma*||offset||^2
              + lambda_adv_adaptive * (-E[D(refined | partial)])
    L_critic  = E[D(refined)] - E[D(real)] + lambda_gp * R1     (n_critic steps)

Warm-start example:
    python training/train_residual_gan.py \
      --resume_vae checkpoints/vae_gan_abl_v2/recon_only/best_vae_gan.pth \
      --data_path /home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset \
      --gt_subdir ground_truth --conditional_critic --adaptive_adv
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

from data.semantickitti import SemanticKITTI, collate_fn
from models.point_cloud_vae import PointCloudVAE
from models.critic_token import TokenCritic, r1_penalty
from models.residual_refine import ResidualRefiner
from models.refinement_net import chamfer_distance
from training.train_vae_gan import (
    normalize_points, sample_to_n, build_norm_batch, build_ctx_batch,
    lambda_adv_effective,
)
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 CD-leashed residual refinement")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--resume_vae", type=str, required=True,
                   help="Frozen Stage-1 VAE checkpoint (CD-optimal, e.g. recon_only best).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    # refiner
    p.add_argument("--refiner_blocks", type=int, default=2)
    p.add_argument("--max_offset", type=float, default=0.1,
                   help="Bound on per-point offset in normalised space (CD leash).")
    p.add_argument("--gamma_offset", type=float, default=1.0,
                   help="Weight of the ||offset||^2 penalty (CD leash).")
    p.add_argument("--lr_gen", type=float, default=3e-4)

    # critic / adversarial
    p.add_argument("--critic_tokens", type=int, default=32)
    p.add_argument("--critic_dim", type=int, default=256)
    p.add_argument("--critic_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--conditional_critic", action="store_true", default=False)
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=3)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_adv_ramp_epochs", type=int, default=3)
    p.add_argument("--disc_warmup_epochs", type=int, default=1)
    p.add_argument("--adaptive_adv", action="store_true", default=False)
    p.add_argument("--adaptive_max", type=float, default=1e4)
    p.add_argument("--adaptive_scale", type=float, default=0.2,
                   help="Sub-unity multiplier on the adaptive ceiling (keep the "
                        "orthogonal adversarial push small relative to recon).")
    p.add_argument("--eps_drift", type=float, default=0.0,
                   help="Epsilon-drift penalty eps*E[D(real)^2] on the critic "
                        "(PGGAN/StyleGAN). Keeps the R1 critic's output scale bounded "
                        "so adv stays O(1) and the adaptive weight does not collapse.")

    # VAE arch (must match the frozen checkpoint)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_dec_blocks", type=int, default=5)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    p.add_argument("--output_dir", type=str, default="checkpoints/residual_gan")
    p.add_argument("--log_dir", type=str, default="logs/residual_gan")
    p.add_argument("--save_freq", type=int, default=5)
    return p.parse_args()


def encode_base_tokens(vae, pts_n, deterministic):
    """Frozen Stage-1: returns base points (K,3) and latent tokens (L,D_int)."""
    mu, logvar = vae.encode(pts_n)
    z = mu if deterministic else vae.reparameterize(mu, logvar)
    base = vae.decode(z)
    tokens = vae.token_up(z.view(vae.num_latent_tokens, vae.token_dim))
    return base, tokens


def train_epoch(vae, refiner, disc, opt_g, opt_d, loader, epoch, args, writer, device):
    refiner.train(); disc.train()
    use_adv = epoch >= args.disc_warmup_epochs
    lae = lambda_adv_effective(epoch, args)
    K = args.num_decoded_points
    S = dict(cd=0.0, off=0.0, adv=0.0, disc=0.0, adapt_w=0.0, cos=0.0)
    cos_n = 0; nb = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        pts_list, real_batch, norms = build_norm_batch(
            cc, cb, bsz, True, True, K, device)
        ctx = build_ctx_batch(batch["partial_coord"], batch["partial_batch"],
                              norms, bsz, K, device) if args.conditional_critic else None

        def dfwd(x):
            return disc(x, ctx) if ctx is not None else disc(x)

        # ---- critic steps ----
        if use_adv:
            disc.requires_grad_(True); refiner.requires_grad_(False)
            dacc = 0.0
            for _ in range(args.n_critic):
                with torch.no_grad():
                    refined = []
                    for p in pts_list:
                        base, tok = encode_base_tokens(vae, p, deterministic=False)
                        r, _ = refiner(base, tok)
                        refined.append(r)
                    fake = torch.stack(refined)
                opt_d.zero_grad()
                gp = r1_penalty(disc, real_batch, ctx=ctx)
                rs = dfwd(real_batch)
                d_loss = dfwd(fake.detach()).mean() - rs.mean() + args.lambda_gp * gp
                if args.eps_drift > 0:
                    d_loss = d_loss + args.eps_drift * rs.pow(2).mean()
                d_loss.backward(); opt_d.step(); dacc += d_loss.item()
            S["disc"] += dacc / args.n_critic
            disc.requires_grad_(False); refiner.requires_grad_(True)

        # ---- refiner step ----
        opt_g.zero_grad()
        refined, offs = [], []
        for p in pts_list:
            base, tok = encode_base_tokens(vae, p, deterministic=False)
            r, o = refiner(base, tok)
            refined.append(r); offs.append(o)
        cd = torch.stack([chamfer_distance(refined[b], pts_list[b]) for b in range(bsz)]).mean()
        off = torch.stack([(o ** 2).sum(-1).mean() for o in offs]).mean()
        g_loss = cd + args.gamma_offset * off

        adapt_w_val = 0.0; cos_val = None
        if use_adv:
            fake_g = torch.stack(refined)
            adv = -dfwd(fake_g).mean()
            # Always log alignment cos(g_recon, g_adv) on the refiner head, even
            # when not adaptive (the adaptive weight collapses for a residual that
            # starts at the CD optimum, where ||g_recon|| ~ 0; fixed lambda_adv +
            # the offset penalty are the CD leash instead).
            last = refiner.head.weight
            g_rec = torch.autograd.grad(cd, last, retain_graph=True)[0]
            g_adv = torch.autograd.grad(adv, last, retain_graph=True)[0]
            cos_val = torch.nn.functional.cosine_similarity(
                g_rec.flatten(), g_adv.flatten(), dim=0).item()
            if args.adaptive_adv:
                aw = (g_rec.norm() / (g_adv.norm() + 1e-8)).clamp(0, args.adaptive_max).detach()
                aw = aw * args.adaptive_scale
                adapt_w_val = aw.item()
            else:
                aw = 1.0
            g_loss = g_loss + lae * aw * adv
            S["adv"] += adv.item(); S["adapt_w"] += adapt_w_val
            if cos_val is not None:
                S["cos"] += cos_val; cos_n += 1

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt_g.step()
        S["cd"] += cd.item(); S["off"] += off.item(); nb += 1

    out = {k: v / max(nb, 1) for k, v in S.items()}
    out["cos"] = (S["cos"] / cos_n) if cos_n else None
    out["lae"] = lae
    return out


@torch.no_grad()
def validate(vae, refiner, loader, device):
    refiner.eval()
    tot = 0.0; n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        for b in range(bsz):
            p, _, _ = normalize_points(cc[cb == b])
            base, tok = encode_base_tokens(vae, p, deterministic=True)
            r, _ = refiner(base, tok)
            tot += chamfer_distance(r, p).item() / bsz
        n += 1
    return tot / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.output_dir); logger.info(args)
    if args.seed is not None:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds_kwargs = {} if args.gt_subdir == "ground_truth" else {"gt_subdir": args.gt_subdir}
    common = dict(root=args.data_path, use_ground_truth_maps=True, use_point_cloud=True,
                  point_max_partial=args.point_max_partial,
                  point_max_complete=args.point_max_complete, **ds_kwargs)
    ds_tr = SemanticKITTI(split="train", augmentation=True, **common)
    ds_va = SemanticKITTI(split="val", augmentation=False, **common)
    tl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    vl = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    vae = PointCloudVAE(latent_dim=args.latent_dim, num_decoded_points=args.num_decoded_points,
                        num_latent_tokens=args.num_latent_tokens, internal_dim=args.internal_dim,
                        num_heads=args.num_heads, num_dec_blocks=args.num_dec_blocks).to(device)
    ck = torch.load(args.resume_vae, map_location=device)
    state = ck.get("vae_state_dict", ck.get("model_state_dict", ck))
    vae.load_state_dict(state); vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded + froze Stage-1 VAE from {args.resume_vae}")

    refiner = ResidualRefiner(dim=args.internal_dim, num_heads=args.num_heads,
                              num_blocks=args.refiner_blocks, max_offset=args.max_offset).to(device)
    disc = TokenCritic(in_dim=3, dim=args.critic_dim, num_tokens=args.critic_tokens,
                       num_heads=args.num_heads, num_layers=args.critic_layers,
                       conditional=args.conditional_critic).to(device)
    logger.info(f"Refiner params: {sum(p.numel() for p in refiner.parameters()):,}  "
                f"Critic params: {sum(p.numel() for p in disc.parameters()):,}")

    opt_g = optim.AdamW(refiner.parameters(), lr=args.lr_gen, weight_decay=1e-4)
    opt_d = optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.0, 0.9))
    writer = SummaryWriter(args.log_dir)
    best = float("inf")

    for epoch in range(args.num_epochs):
        s = train_epoch(vae, refiner, disc, opt_g, opt_d, tl, epoch, args, writer, device)
        va = validate(vae, refiner, vl, device)
        for k in ("cd", "off", "adv", "disc", "adapt_w"):
            writer.add_scalar(f"train/{k}", s[k], epoch)
        if s["cos"] is not None:
            writer.add_scalar("train/cos_recon_adv", s["cos"], epoch)
        writer.add_scalar("val/cd", va, epoch)
        cos_str = f" cos={s['cos']:+.3f}" if s["cos"] is not None else ""
        aw_str = f" aw={s['adapt_w']:.4f}" if s["adapt_w"] else ""
        logger.info(f"Epoch {epoch:3d} | cd={s['cd']:.6f} off={s['off']:.2e} "
                    f"adv={s['adv']:.4f} λ={s['lae']:.3f}{aw_str}{cos_str} val_cd={va:.6f} "
                    f"{'ADV' if epoch>=args.disc_warmup_epochs else 'OFF'}")
        if va < best:
            best = va
            torch.save({"epoch": epoch, "refiner_state_dict": refiner.state_dict(),
                        "disc_state_dict": disc.state_dict(), "val_cd": va,
                        "resume_vae": args.resume_vae},
                       os.path.join(args.output_dir, "best_residual.pth"))
            logger.info(f"  -> saved best (val_cd={best:.6f})")
        if (epoch + 1) % args.save_freq == 0:
            torch.save({"epoch": epoch, "refiner_state_dict": refiner.state_dict(),
                        "disc_state_dict": disc.state_dict(), "val_cd": va,
                        "resume_vae": args.resume_vae},
                       os.path.join(args.output_dir, f"refiner_epoch_{epoch}.pth"))
    writer.close(); logger.info("Stage-2 training complete.")


if __name__ == "__main__":
    main()
