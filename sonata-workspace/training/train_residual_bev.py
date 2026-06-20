"""
BEV-density residual refinement (the CD-orthogonal direct-loss approach).

After the adversary dead-ends (coordinate-space fights CD; latent-space inert),
target the coverage/density gap DIRECTLY: freeze the CD-optimal control VAE,
train a small bounded coordinate residual on the decoded points with
    L = Chamfer(refined, GT) + gamma*||offset||^2 + lambda_bev * JSD_BEV(refined, GT)
The BEV-JSD term optimises the exact coverage metric CD is blind to, and is
invariant to within-cell point position, so it should improve JSD without
pulling points off the surface (CD held by the frozen decoder + offset leash +
the Chamfer anchor). No GAN.

Warm-start:
    python training/train_residual_bev.py \
      --resume_vae checkpoints/vae_gan_abl_v2/recon_only/best_vae_gan.pth \
      --data_path /home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset --gt_subdir ground_truth
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
from models.residual_refine import ResidualRefiner
from models.refinement_net import chamfer_distance
from models.bev_density import bev_jsd_loss
from training.train_vae_gan import normalize_points
from training.train_residual_gan import encode_base_tokens
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="BEV-density residual refinement")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--resume_vae", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--refiner_blocks", type=int, default=2)
    p.add_argument("--max_offset", type=float, default=0.05)
    p.add_argument("--gamma_offset", type=float, default=1.0)
    p.add_argument("--lr_gen", type=float, default=3e-4)

    # BEV-density loss
    p.add_argument("--lambda_bev", type=float, default=1.0)
    p.add_argument("--bev_grid", type=int, default=40)
    p.add_argument("--bev_sigma", type=float, default=0.05)
    p.add_argument("--bev_warmup_epochs", type=int, default=1,
                   help="Epochs of pure recon (Chamfer+offset) before enabling the BEV term.")
    p.add_argument("--bev_ramp_epochs", type=int, default=3)

    # VAE arch (must match the frozen checkpoint)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    p.add_argument("--output_dir", type=str, default="checkpoints/residual_bev")
    p.add_argument("--log_dir", type=str, default="logs/residual_bev")
    p.add_argument("--save_freq", type=int, default=5)
    return p.parse_args()


def bev_weight(epoch, args):
    if epoch < args.bev_warmup_epochs:
        return 0.0
    if args.bev_ramp_epochs <= 0:
        return args.lambda_bev
    frac = (epoch - args.bev_warmup_epochs + 1) / args.bev_ramp_epochs
    return args.lambda_bev * min(1.0, max(0.0, frac))


def train_epoch(vae, refiner, opt_g, loader, epoch, args, writer, device):
    refiner.train()
    lbev = bev_weight(epoch, args)
    use_bev = lbev > 0
    S = dict(cd=0.0, off=0.0, bev=0.0, cos=0.0)
    cos_n = 0; nb = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        pts_list = [normalize_points(cc[cb == b])[0] for b in range(bsz)]

        opt_g.zero_grad()
        refined, offs, bevs = [], [], []
        for p in pts_list:
            base, tok = encode_base_tokens(vae, p, deterministic=False)
            r, o = refiner(base, tok)
            refined.append(r); offs.append(o)
        cd = torch.stack([chamfer_distance(refined[b], pts_list[b]) for b in range(bsz)]).mean()
        off = torch.stack([(o ** 2).sum(-1).mean() for o in offs]).mean()
        g_loss = cd + args.gamma_offset * off

        cos_val = None
        if use_bev:
            bev = torch.stack([
                bev_jsd_loss(refined[b], pts_list[b], grid=args.bev_grid, sigma=args.bev_sigma)
                for b in range(bsz)
            ]).mean()
            last = refiner.head.weight
            g_rec = torch.autograd.grad(cd, last, retain_graph=True)[0]
            g_bev = torch.autograd.grad(bev, last, retain_graph=True)[0]
            cos_val = torch.nn.functional.cosine_similarity(
                g_rec.flatten(), g_bev.flatten(), dim=0).item()
            g_loss = g_loss + lbev * bev
            S["bev"] += bev.item(); S["cos"] += cos_val; cos_n += 1

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt_g.step()
        S["cd"] += cd.item(); S["off"] += off.item(); nb += 1

    out = {k: v / max(nb, 1) for k, v in S.items()}
    out["cos"] = (S["cos"] / cos_n) if cos_n else None
    out["lbev"] = lbev
    return out


@torch.no_grad()
def validate(vae, refiner, loader, args, device):
    refiner.eval()
    tot_cd = 0.0; tot_bev = 0.0; n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        for b in range(bsz):
            p = normalize_points(cc[cb == b])[0]
            base, tok = encode_base_tokens(vae, p, deterministic=True)
            r, _ = refiner(base, tok)
            tot_cd += chamfer_distance(r, p).item() / bsz
            tot_bev += bev_jsd_loss(r, p, grid=args.bev_grid, sigma=args.bev_sigma).item() / bsz
        n += 1
    return tot_cd / max(n, 1), tot_bev / max(n, 1)


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
    vae.load_state_dict(ck.get("vae_state_dict", ck.get("model_state_dict", ck)))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded + froze Stage-1 VAE from {args.resume_vae}")

    refiner = ResidualRefiner(dim=args.internal_dim, num_heads=args.num_heads,
                              num_blocks=args.refiner_blocks, max_offset=args.max_offset).to(device)
    logger.info(f"Refiner params: {sum(p.numel() for p in refiner.parameters()):,}")

    opt_g = optim.AdamW(refiner.parameters(), lr=args.lr_gen, weight_decay=1e-4)
    writer = SummaryWriter(args.log_dir)
    best = float("inf")

    for epoch in range(args.num_epochs):
        s = train_epoch(vae, refiner, opt_g, tl, epoch, args, writer, device)
        va_cd, va_bev = validate(vae, refiner, vl, args, device)
        for k in ("cd", "off", "bev"):
            writer.add_scalar(f"train/{k}", s[k], epoch)
        if s["cos"] is not None:
            writer.add_scalar("train/cos_recon_bev", s["cos"], epoch)
        writer.add_scalar("val/cd", va_cd, epoch)
        writer.add_scalar("val/bev_jsd", va_bev, epoch)
        cos_str = f" cos={s['cos']:+.3f}" if s["cos"] is not None else ""
        logger.info(f"Epoch {epoch:3d} | cd={s['cd']:.6f} off={s['off']:.2e} bev={s['bev']:.5f} "
                    f"lbev={s['lbev']:.3f}{cos_str} val_cd={va_cd:.6f} val_bev={va_bev:.5f} "
                    f"{'BEV' if epoch>=args.bev_warmup_epochs else 'OFF'}")
        ck_out = {"epoch": epoch, "refiner_state_dict": refiner.state_dict(),
                  "val_cd": va_cd, "val_bev": va_bev, "resume_vae": args.resume_vae,
                  "max_offset": args.max_offset, "refiner_blocks": args.refiner_blocks}
        # select on val_bev while CD stays leashed (within 10% of control ~0.0005)
        score = va_bev if va_cd < 0.00055 else va_bev + 1.0
        if score < best:
            best = score
            torch.save(ck_out, os.path.join(args.output_dir, "best_residual.pth"))
            logger.info(f"  -> saved best (val_bev={va_bev:.5f}, val_cd={va_cd:.6f})")
        if (epoch + 1) % args.save_freq == 0:
            torch.save(ck_out, os.path.join(args.output_dir, f"refiner_epoch_{epoch}.pth"))
    writer.close(); logger.info("BEV residual training complete.")


if __name__ == "__main__":
    main()
