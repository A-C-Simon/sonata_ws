#!/usr/bin/env python3
"""
Structured diagnostics for diffusion scene completion (prints only; no loss/arch changes).

Run from repo root, e.g.:
  python scripts/diagnose_completion.py \\
    --input .../velodyne/000050.bin \\
    --checkpoint checkpoints/.../final_model.pth \\
    --gt-path optional_gt.ply
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import (  # noqa: E402
    _subsample_partial_dict,
    apply_completion_profile,
    build_scene_completion_model,
    load_scan,
    prepare_input,
)
from models.diffusion_module import build_shell_coords_single  # noqa: E402
from utils.point_cloud import crop_lidar_radius  # noqa: E402


def _clone_partial(partial: dict) -> dict:
    out = {}
    for k, v in partial.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _mean_nn_dist(pred: torch.Tensor, ref: torch.Tensor, chunk: int = 2048) -> float:
    pred = pred.float()
    ref = ref.float()
    parts = []
    for i in range(0, pred.shape[0], chunk):
        block = pred[i : i + chunk]
        d = torch.cdist(block, ref).min(dim=1).values
        parts.append(d.mean())
    return float(torch.stack(parts).mean().item())


def _make_model_args(args: argparse.Namespace) -> argparse.Namespace:
    qrad = float(args.query_max_radius)
    if args.scene_radius > 0:
        qrad = min(qrad, float(args.scene_radius))
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        encoder_ckpt=args.encoder_ckpt,
        enable_flash=False,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        chamfer_max_points=args.chamfer_max_points,
        query_max_radius=args.query_max_radius,
        query_min_radius=args.query_min_radius,
        scene_radius=args.scene_radius,
        num_query_extra=args.num_query_extra,
        denoising_steps=args.denoising_steps,
        conditioning_mode=str(
            getattr(args, "conditioning_mode", "concat")
        ),
        conditioning_scale=float(
            getattr(args, "conditioning_scale", 1.0)
        ),
    )


def _partial_and_shell(args, scan):
    partial_data, center, _ = prepare_input(scan, voxel_size=args.voxel_size)
    mp = int(args.max_partial_points)
    if mp > 0:
        partial_data = _subsample_partial_dict(partial_data, mp)
    qmax = float(args.query_max_radius)
    if args.scene_radius > 0:
        qmax = min(qmax, float(args.scene_radius))
    shell = build_shell_coords_single(
        partial_data["coord"].float(),
        num_query_extra=int(args.num_query_extra),
        target_shell_total=None,
        max_radius=qmax,
        min_radius=float(args.query_min_radius),
        query_voxel_size=args.query_voxel_size,
        rng=None,
    )
    return partial_data, shell, center


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose completion pipeline (prints only)")
    p.add_argument("--input", type=str, required=True, help="LiDAR .bin / .ply / .pcd")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Optional dense GT cloud (step 6): same crop as --scene-radius",
    )
    p.add_argument("--noise-seed", type=int, default=42)
    p.add_argument("--denoising_steps", type=int, default=20)
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--scene-radius", type=float, default=20.0)
    p.add_argument("--num-query-extra", type=int, default=3000)
    p.add_argument("--max-partial-points", type=int, default=8000)
    p.add_argument("--query-min-radius", type=float, default=3.0)
    p.add_argument("--query-max-radius", type=float, default=20.0)
    p.add_argument("--query-voxel-size", type=float, default=0.15)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--chamfer_max_points", type=int, default=4096)
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument(
        "--conditioning-mode",
        type=str,
        default="concat",
        choices=["concat", "additive", "film"],
        dest="conditioning_mode",
        help="Match checkpoint training",
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
        help="Same as inference.py: guided uses late-step anchor + softer α for QUERY diagnostics.",
    )
    p.add_argument(
        "--anchor-alpha",
        type=float,
        default=1.0,
        dest="anchor_alpha",
        help="Partial-row blend (inference semantics); with guided profile this is overwritten.",
    )
    p.add_argument(
        "--anchor-start-step",
        type=int,
        default=None,
        dest="anchor_start_step",
        help="Optional DDIM threshold for anchoring; overwritten when profile=guided.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    apply_completion_profile(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Data: LiDAR partial ===")
    scan_full = load_scan(args.input)
    if args.scene_radius > 0:
        scan = crop_lidar_radius(scan_full, args.scene_radius)
        print(f"  Crop r≤{args.scene_radius}: {scan.shape[0]} pts")
    else:
        scan = scan_full
    partial_data, shell, _center = _partial_and_shell(args, scan)

    model = build_scene_completion_model(_make_model_args(args), device)
    model.eval()

    common = dict(
        num_steps=args.denoising_steps,
        use_query_points=True,
        shell_coords=shell,
        noise_seed=args.noise_seed,
        anchor_alpha=float(getattr(args, "anchor_alpha", 1.0)),
        anchor_start_step=getattr(args, "anchor_start_step", None),
    )
    aa = common["anchor_start_step"]
    print(
        f"Anchor policy: profile={getattr(args, 'completion_profile', 'strict')}, "
        f"α={common['anchor_alpha']:.3f}, "
        f"anchor_start_step={aa if aa is not None else 'all steps'}"
    )

    print("\n=== STEP 1–2–5: Baseline (diagnostics on) ===")
    with torch.no_grad():
        model.complete_scene(
            _clone_partial(partial_data),
            diagnostics=True,
            **common,
        )

    print("\n=== STEP 3: Condition influence (same noise_seed) ===")
    with torch.no_grad():
        out_cond = model.complete_scene(
            _clone_partial(partial_data),
            diagnostics=False,
            zero_condition=False,
            **common,
        )
        out_zero = model.complete_scene(
            _clone_partial(partial_data),
            diagnostics=False,
            zero_condition=True,
            **common,
        )
    n_partial = int(partial_data["coord"].shape[0])
    diff = (out_cond - out_zero).norm(dim=-1)
    mean_all = float(diff.mean().item())
    mean_partial = (
        float(diff[:n_partial].mean().item()) if n_partial > 0 else float("nan")
    )
    mean_query = (
        float(diff[n_partial:].mean().item())
        if n_partial < diff.shape[0]
        else float("nan")
    )
    print(f"Condition influence L2 (ALL points):     {mean_all:.6f}")
    print(f"Condition influence L2 (QUERY ONLY):       {mean_query:.6f}")
    print(f"Condition influence L2 (PARTIAL / anchor): {mean_partial:.6f}")
    if mean_query < 0.1 and str(getattr(args, "completion_profile", "strict")) == "strict":
        print(
            "  Hint: QUERY influence < 0.1 often means the anchor/graph pins geometry; "
            "retry with --completion-profile guided (or strict + --anchor-start-step / lower --anchor-alpha)."
        )

    print("\n=== STEP 4: Anchor effect (alpha=1 vs 0, same seed) ===")
    common_no_anchor_alpha = {k: v for k, v in common.items() if k != "anchor_alpha"}
    with torch.no_grad():
        out_a1 = model.complete_scene(
            _clone_partial(partial_data),
            diagnostics=False,
            anchor_alpha=1.0,
            **common_no_anchor_alpha,
        )
        out_a0 = model.complete_scene(
            _clone_partial(partial_data),
            diagnostics=False,
            anchor_alpha=0.0,
            **common_no_anchor_alpha,
        )
    diff_a = (out_a1 - out_a0).abs().mean()
    print("Anchor effect:", diff_a.item())

    if args.gt_path:
        print("\n=== STEP 6: GT-as-partial (upper bound sanity) ===")
        gt_full = load_scan(args.gt_path)
        if args.scene_radius > 0:
            gt_full = crop_lidar_radius(gt_full, args.scene_radius)
        partial_gt, shell_gt, _ = _partial_and_shell(args, gt_full)
        gt_t = torch.from_numpy(gt_full.copy()).float().to(device)
        with torch.no_grad():
            out_gt = model.complete_scene(
                _clone_partial(partial_gt),
                diagnostics=False,
                shell_coords=shell_gt,
                num_steps=args.denoising_steps,
                use_query_points=True,
                noise_seed=args.noise_seed,
            )
        mnn = _mean_nn_dist(out_gt, gt_t)
        print("Mean NN distance completion → GT points:", mnn)
    else:
        print("\n=== STEP 6: skipped (--gt-path not set) ===")

    print("\n--- Interpretation (heuristic) ---")
    print(
        "  • COND std ~ 0 → dead conditioning.\n"
        "  • Diffusion movement ~ 0 → sampler not moving x_t.\n"
        "  • Condition influence: use QUERY ONLY (per-point L2); ALL is diluted by anchor.\n"
        "  • Anchor effect huge vs tiny → anchor dominates / unused.\n"
        "  • STEP 6: high NN dist with GT-as-input → underfitting or broken dynamics."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
