#!/usr/bin/env python3
"""
Per-term generator gradient-norm diagnostic for the VAE-GAN.

Tests §2 mechanisms #2 (gradient-scale mismatch) and #3 (feature-matching
dominance) directly: decomposes the generator's gradient into its
reconstruction (Chamfer), adversarial (Wasserstein), and feature-matching
components and reports the L2 norm of each on the *generator* parameters —
both over all VAE params and over the last decoder layer (output_head), which
is what VQGAN-style adaptive weighting uses.

Run from the workspace root. Read-only w.r.t. checkpoints.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from torch.utils.data import DataLoader
from models.discriminator import (
    MultiScalePointDiscriminator,
    feature_matching_loss,
    gradient_penalty,
)
from models.point_cloud_vae import PointCloudVAE, kl_divergence
from models.refinement_net import chamfer_distance


def normalize_points(pts, center=True, scale=True):
    centroid = pts.mean(dim=0) if center else torch.zeros(3, device=pts.device)
    pts_c = pts - centroid
    s = pts_c.abs().max().clamp(min=1e-6) if scale else torch.ones(1, device=pts.device)
    return pts_c / s, centroid, s


def sample_to_n(pts, n):
    N = pts.size(0)
    if N == n:
        return pts
    if N > n:
        return pts[torch.randperm(N, device=pts.device)[:n]]
    repeats = (n + N - 1) // N
    return pts.repeat(repeats, 1)[:n]


def grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += p.grad.detach().pow(2).sum().item()
    return tot ** 0.5


def grad_vec(params):
    return torch.cat([
        (p.grad.detach().flatten() if p.grad is not None
         else torch.zeros(p.numel(), device=p.device))
        for p in params
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset")
    ap.add_argument("--gt_subdir", default="ground_truth_v1")
    ap.add_argument("--vae_ckpt", default="checkpoints/vae_gan_v1gt_ramp_scratch_20260528_2244/checkpoint_epoch_29.pth",
                    help="checkpoint providing live VAE weights (the operating point)")
    ap.add_argument("--vae_key", default="vae_state_dict")
    ap.add_argument("--disc_ckpt", default="checkpoints/vae_gan_v1gt_ramp_scratch_20260528_2244/checkpoint_epoch_29.pth")
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--beta_kl", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    vae = PointCloudVAE(latent_dim=1024, num_decoded_points=8000,
                        num_latent_tokens=32, internal_dim=256,
                        num_heads=4, num_dec_blocks=5).to(device)
    ck = torch.load(args.vae_ckpt, map_location=device)
    state = ck.get(args.vae_key, ck.get("model_state_dict", ck))
    vae.load_state_dict(state)
    vae.train()

    disc = MultiScalePointDiscriminator(in_dim=3).to(device)
    dck = torch.load(args.disc_ckpt, map_location=device)
    disc.load_state_dict(dck["disc_state_dict"])
    disc.eval()
    for p in disc.parameters():
        p.requires_grad_(False)

    ds = SemanticKITTI(root=args.data_path, split="val",
                       use_ground_truth_maps=True, augmentation=False,
                       use_point_cloud=True, point_max_partial=20000,
                       point_max_complete=8000, gt_subdir=args.gt_subdir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    head_params = list(vae.output_head.parameters())
    all_params = list(vae.parameters())
    acc = {k: [] for k in ["cd", "adv", "fm", "kl",
                            "cd_head", "adv_head", "fm_head",
                            "cd_val", "adv_val", "fm_val",
                            "cos_cd_adv", "cos_cd_fm"]}

    seen = 0
    for batch in loader:
        if seen >= args.n_batches:
            break
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc, cb = batch["complete_coord"], batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        pts_list = []
        for b in range(bsz):
            p, _, _ = normalize_points(cc[cb == b])
            pts_list.append(p)
        real_batch = torch.stack([sample_to_n(p, args.K) for p in pts_list])

        # forward (shared)
        recon_list, mu_list, lv_list = [], [], []
        for p in pts_list:
            mu, lv = vae.encode(p)
            z = vae.reparameterize(mu, lv)
            recon_list.append(vae.decode(z))
            mu_list.append(mu); lv_list.append(lv)
        fake_batch = torch.stack(recon_list)

        cd = torch.stack([chamfer_distance(recon_list[b], pts_list[b]) for b in range(bsz)]).mean()
        kl = kl_divergence(torch.stack(mu_list), torch.stack(lv_list))
        fake_score, fake_feats = disc(fake_batch, return_features=True)
        with torch.no_grad():
            _, real_feats = disc(real_batch, return_features=True)
        adv = -fake_score.mean()
        fm = feature_matching_loss(real_feats, fake_feats)

        gvecs = {}
        for name, term in [("cd", cd), ("adv", adv), ("fm", fm), ("kl", kl)]:
            vae.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)
            if name != "kl":
                acc[name + "_head"].append(grad_norm(head_params))
                acc[name + "_val"].append(term.item())
                gvecs[name] = grad_vec(all_params)
            acc[name].append(grad_norm(vae.parameters()))
        cos = torch.nn.functional.cosine_similarity
        acc["cos_cd_adv"].append(cos(gvecs["cd"], gvecs["adv"], dim=0).item())
        acc["cos_cd_fm"].append(cos(gvecs["cd"], gvecs["fm"], dim=0).item())
        vae.zero_grad(set_to_none=True)
        seen += 1

    def stat(x):
        a = np.array(x)
        return f"{a.mean():.6g} ± {a.std():.2g}"

    print("=" * 78)
    print(f"  Per-term generator gradient norms  ({seen} val batches, bs={args.batch_size})")
    print(f"  vae_ckpt={args.vae_ckpt}")
    print(f"  disc_ckpt={args.disc_ckpt}  gt={args.gt_subdir}")
    print("=" * 78)
    print("  term      loss_value        ||grad|| (all VAE params)   ||grad|| (output_head)")
    for name in ["cd", "adv", "fm"]:
        print(f"  {name:8s}  {stat(acc[name+'_val']):18s}  {stat(acc[name]):26s}  {stat(acc[name+'_head'])}")
    print(f"  kl        {'-':18s}  {stat(acc['kl']):26s}")

    gcd = np.mean(acc["cd"]); gadv = np.mean(acc["adv"]); gfm = np.mean(acc["fm"])
    hcd = np.mean(acc["cd_head"]); hadv = np.mean(acc["adv_head"]); hfm = np.mean(acc["fm_head"])
    print("-" * 78)
    print("  RAW grad-norm ratios (all VAE params):")
    print(f"    ||g_adv|| / ||g_cd||  = {gadv/gcd:.3g}")
    print(f"    ||g_fm||  / ||g_cd||  = {gfm/gcd:.3g}")
    print("  Effective contribution at the shipped weights (lambda_adv=0.01, lambda_fm=1.0):")
    print(f"    0.01*||g_adv|| / ||g_cd|| = {0.01*gadv/gcd:.3g}")
    print(f"    1.0 *||g_fm||  / ||g_cd|| = {1.0*gfm/gcd:.3g}")
    print(f"    (lambda_fm=10 default):  10*||g_fm||/||g_cd|| = {10*gfm/gcd:.3g}")
    print("  VQGAN-style adaptive lambda on output_head: ||g_cd|| / (||g_adv||+1e-8):")
    print(f"    lambda_adaptive = {hcd/(hadv+1e-8):.3g}   (vs shipped 0.01)")
    print("-" * 78)
    print("  Gradient alignment (cosine on full VAE grad vector):")
    print(f"    cos(g_cd, g_adv) = {stat(acc['cos_cd_adv'])}")
    print(f"    cos(g_cd, g_fm)  = {stat(acc['cos_cd_fm'])}")
    print("  (>0 aligned: adv helps CD; ~0 orthogonal; <0 conflict: adv trades against CD)")


if __name__ == "__main__":
    main()
