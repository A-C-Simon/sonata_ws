#!/usr/bin/env python3
"""
Run scene completion on the same test split as training (e.g. seq 03 time tail),
save PLY per scan: partial (voxelized), model output, GT map — for visual comparison.

Example (local 20 m scene: crop LiDAR + GT, query grid capped to that radius):
  python scripts/run_inference_test_split.py \\
    --data_path /workspace/dataset/SemanticKITTI/dataset \\
    --gt_variant boost_v2 \\
    --split_preset project \\
    --sequences 03 \\
    --intra_seq_val_fraction 0.15 \\
    --intra_seq_test_fraction 0.15 \\
    --checkpoint checkpoints/diffusion_seq03_tvt_hq/best_model.pth \\
    --output_dir outputs/infer_seq03_local20 \\
    --scene-radius 20 \\
    --query-max-radius 20 \\
    --min-total-vs-partial 2.0 \\
    --voxel_size 0.1 \\
    --max_points 8000

  Use ``--scene-radius 0`` for no crop (legacy full-range LiDAR).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.semantickitti import SemanticKITTI
from inference import (
    apply_completion_profile,
    build_scene_completion_model,
    complete_scene,
    load_scan,
    prepare_input,
    save_pointcloud,
)
from utils.point_cloud import crop_lidar_radius
from models.diffusion_module import (
    DEFAULT_TRAIN_MAX_POINTS,
    QUERY_MAX_RADIUS,
    QUERY_MIN_RADIUS,
    build_shell_coords_single,
)


def _apply_gt_variant(args) -> None:
    v = args.gt_variant
    if v == "custom":
        return
    if v == "boost_v2":
        args.gt_subdir = "ground_truth_v2"
        args.gt_name_suffix = "_v2"
    else:
        args.gt_subdir = "ground_truth"
        args.gt_name_suffix = ""


def _parse_sequences(s: str | None):
    if not s or not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _subsample_partial_dict(partial: dict, max_points: int, rng: np.random.Generator) -> dict:
    n = int(partial["coord"].shape[0])
    if n <= max_points:
        return partial
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    idx_t = torch.from_numpy(idx.astype(np.int64))
    out = {}
    for k, v in partial.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v[idx_t].contiguous()
        else:
            out[k] = v
    out["batch"] = torch.zeros(out["coord"].shape[0], dtype=torch.long)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference on SemanticKITTI test split")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument(
        "--gt_variant",
        type=str,
        default="boost_v2",
        choices=["map_from_scans", "boost_v2", "custom"],
    )
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--gt_name_suffix", type=str, default="")
    p.add_argument("--split_preset", type=str, default="project")
    p.add_argument("--sequences", type=str, default="03")
    p.add_argument("--intra_seq_val_fraction", type=float, default=0.15)
    p.add_argument("--intra_seq_test_fraction", type=float, default=0.15)
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument(
        "--max_points",
        type=int,
        default=DEFAULT_TRAIN_MAX_POINTS,
        help="Cap partial voxels (match training --max_points)",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--denoising_steps", type=int, default=20)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "sigmoid"],
        help="Must match training",
    )
    p.add_argument(
        "--chamfer_max_points",
        type=int,
        default=4096,
        help="Must match training --chamfer_max_points",
    )
    p.add_argument("--max_scans", type=int, default=0, help="0 = all test scans")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-query-points",
        action="store_true",
        help="Denoise partial coords only (no query grid)",
    )
    p.add_argument(
        "--scene-radius",
        type=float,
        default=0.0,
        help=(
            "Crop LiDAR (and GT used for target / optional PLY) to this radius (m) "
            "from sensor origin; 0 = no crop (default). Use e.g. 20 for local scene."
        ),
    )
    p.add_argument(
        "--save-full-gt",
        action="store_true",
        help="With --scene-radius>0, still write *_gt.ply as full map (default: save cropped GT)",
    )
    p.add_argument("--query-max-radius", type=float, default=QUERY_MAX_RADIUS)
    p.add_argument(
        "--query-min-radius",
        type=float,
        default=QUERY_MIN_RADIUS,
        dest="query_min_radius",
        help="Match training --query_min_radius (synthetic query xy lower bound)",
    )
    p.add_argument("--query-voxel-size", type=float, default=0.15)
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--enable_flash", action="store_true")
    p.add_argument(
        "--conditioning-mode",
        type=str,
        default="concat",
        choices=["concat", "additive", "film"],
        dest="conditioning_mode",
        help="Match training checkpoint",
    )
    p.add_argument(
        "--conditioning-scale",
        type=float,
        default=1.0,
        dest="conditioning_scale",
    )
    p.add_argument(
        "--completion-profile",
        type=str,
        default="strict",
        choices=["strict", "guided"],
        dest="completion_profile",
        help="Same as inference.py (guided: softer late anchor for conditioning-friendly completion).",
    )
    p.add_argument(
        "--anchor-alpha",
        type=float,
        default=1.0,
        dest="anchor_alpha",
        help="Partial-row anchor blend (see inference.py); overridden when --completion-profile guided",
    )
    p.add_argument(
        "--anchor-start-step",
        type=int,
        default=None,
        dest="anchor_start_step",
        help="Optional: anchor only when DDIM t_step <= this (see inference.py)",
    )
    p.add_argument(
        "--num-query-extra",
        type=int,
        default=DEFAULT_TRAIN_MAX_POINTS,
        dest="num_query_extra",
        help="Shell size = N_partial + this (same as training --num-query-extra).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    apply_completion_profile(args)
    _apply_gt_variant(args)
    seq_ids = _parse_sequences(args.sequences)
    if args.intra_seq_test_fraction > 0:
        if not seq_ids or len(seq_ids) != 1:
            raise ValueError("intra_seq test split requires exactly one --sequences id")

    ds_kw = dict(
        root=args.data_path,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        use_ground_truth_maps=True,
        use_precomputed=False,
        augmentation=False,
        gt_subdir=args.gt_subdir,
        gt_name_suffix=args.gt_name_suffix,
        split_preset=args.split_preset,
        sequence_ids=seq_ids,
        intra_seq_val_fraction=args.intra_seq_val_fraction,
        intra_seq_test_fraction=args.intra_seq_test_fraction,
    )
    test_ds = SemanticKITTI(split="test", **ds_kw)
    n = len(test_ds)
    if n == 0:
        raise RuntimeError("Test dataset is empty (check paths and precomputed filter).")
    limit = n if args.max_scans <= 0 else min(n, args.max_scans)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, test scans: {n}, running: {limit}")
    model = build_scene_completion_model(args, device)
    rng = np.random.default_rng(args.seed)

    def _short_stem(s: str, m: int = 26) -> str:
        return s if len(s) <= m else s[: m - 1] + "…"

    pbar = tqdm(
        range(limit),
        desc="Inference",
        unit="scan",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for i in pbar:
        scan_path = test_ds.scan_files[i]
        gt_path = test_ds.gt_map_files[i]
        stem = os.path.splitext(os.path.basename(scan_path))[0]
        pbar.set_postfix(scan=_short_stem(stem), refresh=True)

        scan = load_scan(scan_path)
        if args.scene_radius > 0:
            scan = crop_lidar_radius(scan, args.scene_radius)
        partial_cpu, center, _ = prepare_input(scan, voxel_size=args.voxel_size)
        partial_cpu = _subsample_partial_dict(partial_cpu, args.max_points, rng)
        n_partial = int(partial_cpu["coord"].shape[0])

        use_query = not args.no_query_points
        qmax = args.query_max_radius
        if args.scene_radius > 0:
            qmax = min(qmax, args.scene_radius)

        if use_query:
            k = int(args.num_query_extra)
            if k < 0:
                raise ValueError("--num-query-extra must be >= 0")
            target_total = n_partial + k
        else:
            target_total = n_partial

        partial = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in partial_cpu.items()}
        cs_kw = dict(
            anchor_alpha=float(getattr(args, "anchor_alpha", 1.0)),
            anchor_start_step=getattr(args, "anchor_start_step", None),
        )
        if use_query:
            shell_t = build_shell_coords_single(
                partial["coord"].float(),
                num_query_extra=int(args.num_query_extra),
                target_shell_total=None,
                max_radius=float(qmax),
                min_radius=float(args.query_min_radius),
                query_voxel_size=float(args.query_voxel_size),
                rng=None,
            )
            cs_kw["shell_coords"] = shell_t
        else:
            cs_kw["shell_coords"] = None
        completed = complete_scene(
            model,
            partial,
            num_steps=args.denoising_steps,
            use_query_points=use_query,
            **cs_kw,
        )
        completed_world = completed + center

        partial_world = partial_cpu["coord"].numpy() + center
        save_pointcloud(
            os.path.join(args.output_dir, f"{stem}_partial.ply"),
            partial_world,
        )
        save_pointcloud(
            os.path.join(args.output_dir, f"{stem}_completed.ply"),
            completed_world,
        )
        gt_world = np.load(gt_path)["points"].astype(np.float32)
        if args.scene_radius > 0 and not args.save_full_gt:
            gt_world = crop_lidar_radius(gt_world, args.scene_radius)
        save_pointcloud(
            os.path.join(args.output_dir, f"{stem}_gt.ply"),
            gt_world,
        )
        pbar.set_postfix(
            scan=_short_stem(stem),
            Np=n_partial,
            tgt=target_total,
        )

    print(f"Done. {limit} scans → {args.output_dir}")


if __name__ == "__main__":
    main()
