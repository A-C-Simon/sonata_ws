#!/usr/bin/env python3
"""180k-output replication of the RA-L Table II kdtree row.

Same fair deployment protocol as run_scaffoldfree_fair_finetuned.py
(ego-bbox duplicated-LiDAR scaffold, NO GT info, single-step x0 at t=200,
GT+/-1m post-crop, LiDiff-style scipy-KDTree squared CD, 50 stride-80 frames),
with ONE change: the scaffold keeps 180,000 points instead of 20,000, so the
model emits a 180k completion (matching the published baselines' output size).

The denoiser runs TILE-WISE over the 180k scaffold in shuffled 20k chunks --
identical to its training/inference context size (in-distribution), with
conditioning features interpolated per-point exactly as in the 20k protocol.

Runs all 6 fine-tune seeds (42-47), epoch-2 selection, mirroring the paper row.
Also reproduces the 20k kdtree row for seed 42 as a sanity anchor.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))

import run_scaffoldfree_fair_finetuned as base  # reuse all helpers

OUT_PTS = 180000
CHUNK = 20000
NUM_FRAMES = 50
RESULTS_DIR = WORK_DIR / "results" / "jul02_night"
SUMMARY_PATH = Path.home() / "RAL_OVERNIGHT_SUMMARY.txt"

SEED_CKPTS = {
    42: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold" / "epoch_2.pth",
    43: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold_seed43" / "epoch_2.pth",
    44: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold_seed44" / "epoch_2.pth",
    45: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold_seed45" / "epoch_2.pth",
    46: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold_seed46" / "epoch_2.pth",
    47: WORK_DIR / "checkpoints" / "diffusion_v2gt_finetune_mixed_scaffold_seed47" / "epoch_2.pth",
}


@torch.no_grad()
def run_x0_chunked(model, point_dict, target_np, t_val=200):
    """Single-step x0 over an arbitrarily large scaffold, in shuffled 20k tiles
    (the model's native context size). Returns (N, 3) prediction, same order
    guarantees are irrelevant downstream (CD is order-invariant)."""
    n = target_np.shape[0]
    perm = np.random.permutation(n)
    preds = np.empty_like(target_np)
    for s in range(0, n, CHUNK):
        idx = perm[s:s + CHUNK]
        tgt = torch.from_numpy(target_np[idx]).float().to(base.DEVICE)
        pred = base.run_x0_single_step(model, point_dict, tgt, t_val=t_val)
        preds[idx] = pred.cpu().numpy()
    return preds


def variant_180k_ego_bbox(model, fr):
    scaffold = base.scaffold_input_duplicated_fair(
        fr, n_dup=10, jitter=0.05, max_pts=OUT_PTS, crop_mode="ego")
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = base.make_point_dict(fr["lidar_coords"])
    pred = run_x0_chunked(model, point_dict, scaffold, t_val=200)
    return pred, scaffold


def variant_20k_sanity(model, fr):
    return base.variant_A_fair_ego_bbox(model, fr)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = base.load_frames(NUM_FRAMES)

    lines = [f"# RA-L overnight 180k-output replication  ({time.strftime('%Y-%m-%d %H:%M')})",
             f"# protocol: ego-bbox dup-LiDAR scaffold, {OUT_PTS} pts (tile-wise x0 in {CHUNK} chunks),",
             "# GT+/-1m crop, kdtree squared CD (60k subsample), 50 stride-80 frames, epoch-2 ckpts",
             ""]
    means = []

    for seed, ckpt in SEED_CKPTS.items():
        print(f"\n########## SEED {seed}  ({ckpt}) ##########")
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = base.build_model()
        base.load_ckpt(model, ckpt)

        variants = [("FT_180k_ego_bbox_kdtree", variant_180k_ego_bbox, "lidiff", "kdtree")]
        if seed == 42:
            variants.append(("FT_20k_sanity_kdtree", variant_20k_sanity, "lidiff", "kdtree"))

        results = {"variants": {}}
        for name, fn, crop_mode, cd_impl in variants:
            np.random.seed(seed)
            torch.manual_seed(seed)
            results["variants"][name] = base.run_variant_all_frames(
                model, frames, name, fn, crop_mode, cd_impl)

        out = RESULTS_DIR / f"teacher_finetuned_str80_180k_seed{seed}.json"
        payload = {**results, "num_frames": NUM_FRAMES, "checkpoint": str(ckpt),
                   "seed": seed, "out_pts": OUT_PTS, "chunk": CHUNK,
                   "lidiff_margin_m": base.LIDIFF_MARGIN}
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[persist] {out}")

        r = results["variants"]["FT_180k_ego_bbox_kdtree"]
        if "pred_cd_mean" in r:
            means.append(r["pred_cd_mean"])
            lines.append(f"seed {seed}: 180k kdtree CD^2 = "
                         f"{r['pred_cd_mean']:.4f} +/- {r['pred_cd_std']:.4f} "
                         f"(n={r['n_frames']}, {r['per_frame_time_s_mean']:.1f}s/frame)")
        else:
            lines.append(f"seed {seed}: ERROR {r.get('error')}")
        if seed == 42 and "FT_20k_sanity_kdtree" in results["variants"]:
            s = results["variants"]["FT_20k_sanity_kdtree"]
            if "pred_cd_mean" in s:
                lines.append(f"seed 42 SANITY 20k kdtree = {s['pred_cd_mean']:.4f} "
                             f"(paper per-seed value: 0.727)")

        del model
        torch.cuda.empty_cache()

        # persist summary incrementally
        with open(SUMMARY_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    if means:
        lines.append("")
        lines.append(f"ACROSS-SEED 180k: mean {np.mean(means):.4f} +/- {np.std(means, ddof=1):.4f} "
                     f"(n_seeds={len(means)})  |  paper 20k row: 0.968 +/- 0.194")
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
