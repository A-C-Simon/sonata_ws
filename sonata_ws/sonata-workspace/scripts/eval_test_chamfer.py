#!/usr/bin/env python3
"""
Chamfer vs GT on N scans from project test split (sequence 10 by default).

Matches the seq02 report protocol: crop GT with --scene-radius, subsample GT to
--gt-sub points (fixed seed per scan), full prediction cloud.
For each scan, evaluates multiple DDIM denoising step counts (`d20/d50/d100`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.semantickitti import SemanticKITTI
from inference import (
    build_scene_completion_model,
    load_scan,
    prepare_input,
    run_completion,
    _subsample_partial_dict,
)
from models.diffusion_module import (
    DEFAULT_TRAIN_MAX_POINTS,
    QUERY_MAX_RADIUS,
    QUERY_MIN_RADIUS,
    build_shell_coords_single,
)
from utils.point_cloud import crop_lidar_radius


def chamfer_symmetric(
    pred: np.ndarray,
    gt_cropped: np.ndarray,
    gt_sub: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    gt = np.asarray(gt_cropped, dtype=np.float64)
    if gt.shape[0] > gt_sub:
        idx = rng.choice(gt.shape[0], size=gt_sub, replace=False)
        gts = gt[idx]
    else:
        gts = gt
    pr = np.asarray(pred, dtype=np.float64)
    tb = cKDTree(gts)
    ta = cKDTree(pr)
    d_ab, _ = tb.query(pr, k=1)
    d_ba, _ = ta.query(gts, k=1)
    l2_sq = float(0.5 * ((d_ab**2).mean() + (d_ba**2).mean()))
    l2 = float(0.5 * (d_ab.mean() + d_ba.mean()))
    return l2_sq, l2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chamfer eval on test split (seq 10)")
    p.add_argument(
        "--data_path",
        type=str,
        default="/workspace/dataset/SemanticKITTI/dataset",
        help="SemanticKITTI root (sequences/, ground_truth/)",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="best_model.pth or other .pth")
    p.add_argument("--num-scans", type=int, default=100, dest="num_scans", help="How many test scans to evaluate")
    p.add_argument("--seed", type=int, default=42, help="Base seed for scan selection + GT subsample")
    p.add_argument("--scene-radius", type=float, default=20.0, dest="scene_radius")
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--max-partial-points", type=int, default=DEFAULT_TRAIN_MAX_POINTS, dest="max_partial_points")
    p.add_argument(
        "--num-query-extra",
        type=int,
        default=DEFAULT_TRAIN_MAX_POINTS,
        dest="num_query_extra",
    )
    p.add_argument("--query-max-radius", type=float, default=QUERY_MAX_RADIUS, dest="query_max_radius")
    p.add_argument(
        "--query-min-radius",
        type=float,
        default=QUERY_MIN_RADIUS,
        dest="query_min_radius",
        help="Match training --query_min_radius",
    )
    p.add_argument("--query-voxel-size", type=float, default=0.15, dest="query_voxel_size")
    p.add_argument(
        "--denoising-steps",
        type=str,
        default="20,50,100",
        dest="denoising_steps",
        help="Comma-separated DDIM step counts to evaluate (e.g. '20,50,100')",
    )
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--chamfer_max_points", type=int, default=4096)
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--enable_flash", action="store_true")
    p.add_argument(
        "--conditioning-mode",
        type=str,
        default="concat",
        choices=["concat", "additive", "film"],
        dest="conditioning_mode",
    )
    p.add_argument(
        "--conditioning-scale", type=float, default=1.0, dest="conditioning_scale"
    )
    p.add_argument("--amp-inference", action="store_true", dest="amp_inference")
    p.add_argument("--gt-sub", type=int, default=12000, dest="gt_sub", help="GT subsample size for Chamfer")
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write aggregate metrics + per-scan list here",
    )
    return p.parse_args()


def _parse_denoising_steps(s: str) -> List[int]:
    steps = [x.strip() for x in str(s).split(",") if x.strip()]
    if not steps:
        raise ValueError("Empty --denoising-steps")
    out: List[int] = []
    for x in steps:
        v = int(x)
        if v <= 0:
            raise ValueError(f"Invalid denoising step count: {v}")
        out.append(v)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    test_ds = SemanticKITTI(
        root=args.data_path,
        split="test",
        voxel_size=args.voxel_size,
        max_points=args.max_partial_points,
        use_ground_truth_maps=True,
        use_precomputed=False,
        augmentation=False,
        split_preset="project",
        scene_radius=args.scene_radius,
    )
    n_total = len(test_ds)
    if n_total == 0:
        raise RuntimeError("Test dataset empty (check paths and GT).")
    rng = np.random.default_rng(args.seed)
    n_eval = min(args.num_scans, n_total)
    indices = rng.choice(n_total, size=n_eval, replace=False)
    indices = np.sort(indices)

    denoising_steps = _parse_denoising_steps(args.denoising_steps)

    ns = argparse.Namespace(
        checkpoint=args.checkpoint,
        encoder_ckpt=args.encoder_ckpt,
        enable_flash=args.enable_flash,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        # Only used when `num_steps` is None. We override per-evaluation variant anyway.
        denoising_steps=max(denoising_steps),
        chamfer_max_points=args.chamfer_max_points,
        num_query_extra=args.num_query_extra,
        query_max_radius=args.query_max_radius,
        query_min_radius=args.query_min_radius,
        scene_radius=args.scene_radius,
        conditioning_mode=args.conditioning_mode,
        conditioning_scale=args.conditioning_scale,
    )
    print(f"Loading model from {args.checkpoint} …")
    t0 = time.perf_counter()
    model = build_scene_completion_model(ns, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Model load: {time.perf_counter() - t0:.1f}s")

    qmax = min(float(args.query_max_radius), float(args.scene_radius))
    per_step_l2sq: dict[int, List[float]] = {s: [] for s in denoising_steps}
    per_step_l2: dict[int, List[float]] = {s: [] for s in denoising_steps}
    stems: List[str] = []
    per_scan: List[dict] = []

    for j, i in enumerate(indices):
        scan_path = test_ds.scan_files[int(i)]
        gt_path = test_ds.gt_map_files[int(i)]
        stem = os.path.splitext(os.path.basename(scan_path))[0]
        print(f"[{j+1}/{n_eval}] {stem} …", flush=True)

        scan_full = load_scan(scan_path)
        scan = crop_lidar_radius(scan_full, args.scene_radius) if args.scene_radius > 0 else scan_full
        if scan.shape[0] == 0:
            print("  skip: empty after crop")
            continue
        partial_data, center, _ = prepare_input(scan, voxel_size=args.voxel_size)
        if args.max_partial_points > 0:
            partial_data = _subsample_partial_dict(partial_data, args.max_partial_points, seed=args.seed + j)

        shell_coords_cpu = build_shell_coords_single(
            partial_data["coord"].float(),
            num_query_extra=int(args.num_query_extra),
            target_shell_total=None,
            max_radius=qmax,
            min_radius=float(args.query_min_radius),
            query_voxel_size=args.query_voxel_size,
            rng=None,
        )
        rc_kw = dict(anchor_alpha=1.0, shell_coords=shell_coords_cpu)

        # Keep GT subsampling deterministic and scan-specific (matches seq02 seed protocol).
        sub_seed = int(args.seed) + int(j)
        scan_metrics = {"stem": stem, "variants": {}}
        t_inf0 = time.perf_counter()

        gt_raw = np.load(gt_path)["points"].astype(np.float32)
        gt_crop = crop_lidar_radius(gt_raw, args.scene_radius)

        for step in denoising_steps:
            completed = run_completion(
                model,
                partial_data,
                num_steps=step,
                use_query_points=True,
                use_amp=bool(args.amp_inference),
                **rc_kw,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            pred = completed + center

            c2, c1 = chamfer_symmetric(pred, gt_crop, args.gt_sub, sub_seed)
            per_step_l2sq[step].append(c2)
            per_step_l2[step].append(c1)
            scan_metrics["variants"][str(step)] = {"chamfer_m2": c2, "chamfer_m": c1}
            print(f"  d{step}: Chamfer m²={c2:.6f} m={c1:.6f}")

        print(f"  (done in {time.perf_counter() - t_inf0:.1f}s)")
        stems.append(stem)
        per_scan.append(scan_metrics)
    summary = {
        "checkpoint": args.checkpoint,
        "num_scans_evaluated": len(stems),
        "scene_radius_m": args.scene_radius,
        "gt_subsample": args.gt_sub,
        "denoising_steps": denoising_steps,
        "per_step": {},
        "per_scan": per_scan,
    }

    print("\n=== Aggregate (test seq 10) ===")
    table_lines = []
    for step in denoising_steps:
        l2sq_arr = np.array(per_step_l2sq[step], dtype=np.float64)
        l2_arr = np.array(per_step_l2[step], dtype=np.float64)
        c2_mean = float(l2sq_arr.mean()) if len(l2sq_arr) else None
        c2_std = float(l2sq_arr.std()) if len(l2sq_arr) else None
        c1_mean = float(l2_arr.mean()) if len(l2_arr) else None
        c1_std = float(l2_arr.std()) if len(l2_arr) else None
        summary["per_step"][str(step)] = {
            "chamfer_m2_mean": c2_mean,
            "chamfer_m2_std": c2_std,
            "chamfer_m_mean": c1_mean,
            "chamfer_m_std": c1_std,
        }
        table_lines.append((step, c2_mean, c1_mean))

    # Print in the same style as seq02 report table.
    for step, c2_mean, c1_mean in table_lines:
        print(f"d{step}: Chamfer m²={c2_mean:.6f}  Chamfer m={c1_mean:.6f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
