"""
Latent-space adversarial refinement for the VAE-GAN.

Freeze a CD-optimal Stage-1 VAE (encoder AND decoder = the recon_only control).
Encode each GT to its latent tokens mu; a small bounded LatentRefiner nudges the
tokens within the latent manifold; the FROZEN decoder maps them back to points.
A LatentTokenCritic scores token realism (WGAN + R1 + eps-drift). Because the
adversary never touches XYZ and the decoder constrains outputs to its learned
manifold, the adversarial gradient should stop fighting Chamfer (the central
hypothesis: cos(g_recon, g_adv) >= 0, unlike the -0.5..-0.7 of coordinate-space
critics). CD is leashed by the frozen decoder + a latent-offset penalty.

    refined_z = mu + max_dz * tanh(refiner(mu))          (in normalised latent space)
    decoded   = frozen_decode(denorm(refined_z))
    L_refiner = Chamfer(decoded, GT) + gamma_z*||dz||^2 + lambda_adv*(-E[D(refined_z)])
    L_critic  = E[D(refined_z)] - E[D(mu)] + lambda_gp*R1 + eps_drift*E[D(mu)^2]

Warm-start example:
    python training/train_latent_gan.py \
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
from models.latent_critic import LatentTokenCritic, LatentRefiner, r1_penalty_latent
from models.latent_diffusion import LatentNormalizer
from models.refinement_net import chamfer_distance
from training.train_vae_gan import normalize_points, lambda_adv_effective
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Latent-space adversarial refinement")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--resume_vae", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    # refiner / critic
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--refiner_layers", type=int, default=3)
    p.add_argument("--critic_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--max_dz", type=float, default=0.5,
                   help="Bound on per-token latent offset in NORMALISED latent space.")
    p.add_argument("--gamma_z", type=float, default=1.0,
                   help="Weight of the ||dz||^2 latent-offset leash.")
    p.add_argument("--lr_gen", type=float, default=3e-4)
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=2)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--lambda_adv", type=float, default=1e-2)
    p.add_argument("--lambda_adv_ramp_epochs", type=int, default=3)
    p.add_argument("--disc_warmup_epochs", type=int, default=1)
    p.add_argument("--eps_drift", type=float, default=1e-2)

    # VAE arch (must match the frozen checkpoint)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_dec_blocks", type=int, default=5)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    p.add_argument("--output_dir", type=str, default="checkpoints/latent_gan")
    p.add_argument("--log_dir", type=str, default="logs/latent_gan")
    p.add_argument("--save_freq", type=int, default=5)
    return p.parse_args()


def encode_mu(vae, pts_n):
    """Frozen encoder: normalised points -> latent mu (latent_dim,)."""
    mu, _ = vae.encode(pts_n)
    return mu


def train_epoch(vae, refiner, critic, norm, opt_g, opt_d, loader, epoch, args, writer, device):
    refiner.train(); critic.train()
    use_adv = epoch >= args.disc_warmup_epochs
    lae = lambda_adv_effective(epoch, args)
    L, td = args.num_latent_tokens, args.latent_dim // args.num_latent_tokens
    S = dict(cd=0.0, dz=0.0, adv=0.0, disc=0.0, cos=0.0)
    cos_n = 0; nb = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        pts_list = [normalize_points(cc[cb == b])[0] for b in range(bsz)]

        with torch.no_grad():
            mu_flat = torch.stack([encode_mu(vae, p) for p in pts_list])   # (B, latent_dim)
            norm.update(mu_flat)
            mu_n = norm.normalize(mu_flat)                                  # (B, latent_dim)
        mu_n_tok = mu_n.view(bsz, L, td)

        # ---- critic steps ----
        if use_adv:
            critic.requires_grad_(True); refiner.requires_grad_(False)
            dacc = 0.0
            for _ in range(args.n_critic):
                with torch.no_grad():
                    ref_tok, _ = refiner(mu_n_tok)
                opt_d.zero_grad()
                rs = critic(mu_n_tok)
                gp = r1_penalty_latent(critic, mu_n_tok)
                d_loss = critic(ref_tok.detach()).mean() - rs.mean() + args.lambda_gp * gp
                if args.eps_drift > 0:
                    d_loss = d_loss + args.eps_drift * rs.pow(2).mean()
                d_loss.backward(); opt_d.step(); dacc += d_loss.item()
            S["disc"] += dacc / args.n_critic
            critic.requires_grad_(False); refiner.requires_grad_(True)

        # ---- refiner step ----
        opt_g.zero_grad()
        ref_tok, dz = refiner(mu_n_tok)                                    # (B, L, td)
        refined_flat = norm.denormalize(ref_tok.reshape(bsz, args.latent_dim))
        decoded = [vae.decode(refined_flat[b]) for b in range(bsz)]
        cd = torch.stack([chamfer_distance(decoded[b], pts_list[b]) for b in range(bsz)]).mean()
        dzpen = dz.pow(2).sum(-1).mean()
        g_loss = cd + args.gamma_z * dzpen

        cos_val = None
        if use_adv:
            adv = -critic(ref_tok).mean()
            last = refiner.head.weight
            g_rec = torch.autograd.grad(cd, last, retain_graph=True)[0]
            g_adv = torch.autograd.grad(adv, last, retain_graph=True)[0]
            cos_val = torch.nn.functional.cosine_similarity(
                g_rec.flatten(), g_adv.flatten(), dim=0).item()
            g_loss = g_loss + lae * adv
            S["adv"] += adv.item(); S["cos"] += cos_val; cos_n += 1

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt_g.step()
        S["cd"] += cd.item(); S["dz"] += dzpen.item(); nb += 1

    out = {k: v / max(nb, 1) for k, v in S.items()}
    out["cos"] = (S["cos"] / cos_n) if cos_n else None
    out["lae"] = lae
    return out


@torch.no_grad()
def validate(vae, refiner, norm, loader, args, device):
    refiner.eval()
    L, td = args.num_latent_tokens, args.latent_dim // args.num_latent_tokens
    tot = 0.0; n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        for b in range(bsz):
            p = normalize_points(cc[cb == b])[0]
            mu = encode_mu(vae, p)
            mu_n = norm.normalize(mu.unsqueeze(0)).view(1, L, td)
            ref_tok, _ = refiner(mu_n)
            refined_flat = norm.denormalize(ref_tok.reshape(1, args.latent_dim))[0]
            tot += chamfer_distance(vae.decode(refined_flat), p).item() / bsz
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
    vae.load_state_dict(ck.get("vae_state_dict", ck.get("model_state_dict", ck)))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded + froze Stage-1 VAE (encoder+decoder) from {args.resume_vae}")

    td = args.latent_dim // args.num_latent_tokens
    refiner = LatentRefiner(td, dim=args.hidden, num_heads=args.num_heads,
                            num_layers=args.refiner_layers, num_tokens=args.num_latent_tokens,
                            max_dz=args.max_dz).to(device)
    critic = LatentTokenCritic(td, dim=args.hidden, num_heads=args.num_heads,
                               num_layers=args.critic_layers, num_tokens=args.num_latent_tokens).to(device)
    norm = LatentNormalizer(args.latent_dim).to(device)
    logger.info(f"Refiner params: {sum(p.numel() for p in refiner.parameters()):,}  "
                f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    opt_g = optim.AdamW(refiner.parameters(), lr=args.lr_gen, weight_decay=1e-4)
    opt_d = optim.Adam(critic.parameters(), lr=args.lr_disc, betas=(0.0, 0.9))
    writer = SummaryWriter(args.log_dir)
    best = float("inf")

    for epoch in range(args.num_epochs):
        s = train_epoch(vae, refiner, critic, norm, opt_g, opt_d, tl, epoch, args, writer, device)
        va = validate(vae, refiner, norm, vl, args, device)
        for k in ("cd", "dz", "adv", "disc"):
            writer.add_scalar(f"train/{k}", s[k], epoch)
        if s["cos"] is not None:
            writer.add_scalar("train/cos_recon_adv", s["cos"], epoch)
        writer.add_scalar("val/cd", va, epoch)
        cos_str = f" cos={s['cos']:+.3f}" if s["cos"] is not None else ""
        logger.info(f"Epoch {epoch:3d} | cd={s['cd']:.6f} dz={s['dz']:.3e} adv={s['adv']:.4f} "
                    f"lambda={s['lae']:.4f}{cos_str} val_cd={va:.6f} "
                    f"{'ADV' if epoch>=args.disc_warmup_epochs else 'OFF'}")
        ck_out = {"epoch": epoch, "refiner_state_dict": refiner.state_dict(),
                  "critic_state_dict": critic.state_dict(),
                  "norm_state_dict": norm.state_dict(), "val_cd": va,
                  "resume_vae": args.resume_vae}
        if va < best:
            best = va
            torch.save(ck_out, os.path.join(args.output_dir, "best_latent.pth"))
            logger.info(f"  -> saved best (val_cd={best:.6f})")
        if (epoch + 1) % args.save_freq == 0:
            torch.save(ck_out, os.path.join(args.output_dir, f"latent_epoch_{epoch}.pth"))
    writer.close(); logger.info("Latent-GAN training complete.")


if __name__ == "__main__":
    main()
