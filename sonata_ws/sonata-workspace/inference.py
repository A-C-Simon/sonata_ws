"""
Single-scan scene completion CLI.

**Task:** one LiDAR frame → optional **Euclidean crop** from the sensor (``--scene-radius``,
default 20 m, same as ``crop_lidar_radius`` / training ``SemanticKITTI(scene_radius=…)``),
then voxelize and run diffusion. The on-disk scan is never modified; only the in-memory
tensor fed to the model is cropped.

Query layout uses a uniform cylinder: ``r_xy ∈ [0, query-max-radius]``, ``z ∈ [-R,R]``,
capped by ``scene_radius`` when that is enabled for consistency with the local GT crop.

Pipeline: **crop (if radius > 0) → voxelize → build shell via ``build_shell_coords_single``
(same path as training ``build_shell_coords_from_partial_only``) → encoder → diffusion**.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Set, Tuple

import numpy as np
import open3d as o3d
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.diffusion_module import (
    DEFAULT_TRAIN_MAX_POINTS,
    QUERY_MAX_RADIUS,
    QUERY_MIN_RADIUS,
    SceneCompletionDiffusion,
    build_shell_coords_single,
)
from models.sonata_encoder import ConditionalFeatureExtractor, SonataEncoder
from utils.checkpoint import load_checkpoint
from utils.point_cloud import crop_lidar_radius


GUIDED_ANCHOR_ALPHA = 0.65
"""Default partial-row blend when ``--completion-profile guided`` (late-step anchoring)."""


def apply_completion_profile(ns: argparse.Namespace) -> None:
    """Mutate ``anchor_alpha`` / ``anchor_start_step`` for ``guided`` profile; ``strict`` is a no-op."""
    if getattr(ns, "completion_profile", "strict") != "guided":
        return
    ns.anchor_alpha = float(GUIDED_ANCHOR_ALPHA)
    # Anchor only when t_step <= this (t counts down); first ~half of DDIM iters leave partial rows free.
    ns.anchor_start_step = max(0, (int(ns.denoising_steps) - 1) // 2)


def _parse_denoise_snapshot_iters(s: Optional[str]) -> Optional[Set[int]]:
    """Comma-separated ints: 0 = after q_sample; k>=1 = after k-th DDIM iteration."""
    if s is None or not str(s).strip():
        return None
    out: Set[int] = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part, 10)
        if v < 0:
            raise ValueError("--save-denoise-checkpoints values must be >= 0")
        out.add(v)
    return out


def _subsample_partial_dict(
    partial: Dict[str, torch.Tensor],
    max_points: int,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Randomly cap partial voxels (same idea as batch inference script)."""
    n = int(partial["coord"].shape[0])
    if n <= max_points:
        return partial
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:max_points]
    idx = torch.sort(idx).values
    out: Dict[str, torch.Tensor] = {}
    for k, v in partial.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v[idx].contiguous()
        else:
            out[k] = v
    out["batch"] = torch.zeros(out["coord"].shape[0], dtype=torch.long)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scene completion inference (shell + diffusion)")
    p.add_argument("--input", type=str, required=True, help="Scan .bin / .pcd / .ply")
    p.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint .pth")
    p.add_argument("--output", type=str, default="output_completed.ply")
    p.add_argument(
        "--save-partial",
        type=str,
        default=None,
        dest="save_partial",
        help="Optional PLY path: voxelized partial (what the model conditions on), sensor frame.",
    )
    p.add_argument(
        "--denoising_steps",
        type=int,
        default=20,
        help="DDIM reverse steps in complete_scene (typical 10–30)",
    )
    p.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Must match training (same as train_diffusion --num_timesteps)",
    )
    p.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "sigmoid"],
        help="Must match training (same as train_diffusion --schedule)",
    )
    p.add_argument(
        "--chamfer_max_points",
        type=int,
        default=4096,
        help="Must match training --chamfer_max_points (architecture slot; loss may not use it)",
    )
    p.add_argument("--visualize", action="store_true")
    p.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Partial voxel size (match training / GT maps)",
    )
    p.add_argument(
        "--no-query-points",
        action="store_true",
        help="Denoise partial voxels only (no query shell)",
    )
    p.add_argument("--query-max-radius", type=float, default=QUERY_MAX_RADIUS)
    p.add_argument(
        "--query-min-radius",
        type=float,
        default=QUERY_MIN_RADIUS,
        dest="query_min_radius",
        help="Synthetic query r_xy lower bound (m); must match training (default blind-zone cutout)",
    )
    p.add_argument("--query-voxel-size", type=float, default=0.15)
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--enable_flash", action="store_true")
    p.add_argument(
        "--scene-radius",
        type=float,
        default=QUERY_MAX_RADIUS,
        help=(
            "Euclidean distance from sensor (m) to keep before voxelize and diffusion; "
            f"0 disables. Default {QUERY_MAX_RADIUS} matches typical 20 m local scene; "
            "set equal to training --scene-radius. When >0, query max radius is min(..., this)."
        ),
    )
    p.add_argument(
        "--num-query-extra",
        type=int,
        default=DEFAULT_TRAIN_MAX_POINTS,
        dest="num_query_extra",
        help=(
            "Shell size = N_partial + this (same meaning as training --num-query-extra). "
            f"Default {DEFAULT_TRAIN_MAX_POINTS}; set to training value."
        ),
    )
    p.add_argument(
        "--max-partial-points",
        type=int,
        default=DEFAULT_TRAIN_MAX_POINTS,
        help=(
            "After voxelize, cap partial voxels (match training --max_points). "
            f"Default {DEFAULT_TRAIN_MAX_POINTS}. Use 0 to disable (keep all voxels)."
        ),
    )
    p.add_argument(
        "--amp-inference",
        action="store_true",
        help="Run complete_scene under CUDA float16 autocast (faster, slight numeric drift).",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print conditioning / shell / diffusion-movement stats in complete_scene (debug).",
    )
    p.add_argument(
        "--conditioning-mode",
        type=str,
        default="concat",
        choices=["concat", "additive", "film"],
        dest="conditioning_mode",
        help="Denoiser fusion: concat (default) | additive | film (must match training for checkpoints).",
    )
    p.add_argument(
        "--conditioning-scale",
        type=float,
        default=1.0,
        dest="conditioning_scale",
        help="Strength for additive / FiLM conditioning (ignored for concat).",
    )
    p.add_argument(
        "--completion-profile",
        type=str,
        default="strict",
        choices=["strict", "guided"],
        dest="completion_profile",
        help=(
            "strict: hard anchor every step (default; backward compatible). "
            "guided: softer completion — α="
            f"{GUIDED_ANCHOR_ALPHA} and anchor only on late DDIM steps (see --anchor-start-step). "
            "For custom α/schedule use strict + --anchor-alpha / --anchor-start-step."
        ),
    )
    p.add_argument(
        "--anchor-alpha",
        type=float,
        default=1.0,
        dest="anchor_alpha",
        help=(
            "Per DDIM step (when anchor active): partial rows = α·partial + (1-α)·predicted, "
            "or exact partial when α=1.0. Use <1 to let observed voxels move slightly."
        ),
    )
    p.add_argument(
        "--anchor-start-step",
        type=int,
        default=None,
        dest="anchor_start_step",
        help=(
            "If set, anchor applies only when DDIM sub-step index t_step <= this value "
            "(t_step counts down from denoising_steps-1 to 0). Early steps skip anchor so "
            "conditioning can affect the whole shell."
        ),
    )
    p.add_argument(
        "--save-denoise-checkpoints",
        type=str,
        default=None,
        dest="save_denoise_checkpoints",
        help=(
            "Extra PLY snapshots while denoising. Comma-separated indices: 0 = right after "
            "noisy init (q_sample), before first DDIM step; k≥1 = after k-th DDIM iteration. "
            "Files: <output_stem>_denoise_XXX.ply (sensor frame). Final --output is always written."
        ),
    )
    return p.parse_args()


def load_scan(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return scan[:, :3]
    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    raise ValueError(f"Unsupported format: {ext}")


def save_pointcloud(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved point cloud to {path}")


def visualize_comparison(partial: np.ndarray, complete: np.ndarray) -> None:
    p_partial = o3d.geometry.PointCloud()
    p_partial.points = o3d.utility.Vector3dVector(partial)
    p_partial.paint_uniform_color([1.0, 0.0, 0.0])
    p_complete = o3d.geometry.PointCloud()
    p_complete.points = o3d.utility.Vector3dVector(complete)
    p_complete.paint_uniform_color([0.0, 1.0, 0.0])
    p_complete.translate(np.array([50.0, 0.0, 0.0]))
    o3d.visualization.draw_geometries(
        [p_partial, p_complete],
        window_name="Partial (red) vs complete (green)",
        width=1920,
        height=1080,
    )


def prepare_input(
    scan: np.ndarray, voxel_size: float = 0.1
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Voxelize in LiDAR frame (origin = sensor). Returns ``partial_dict``, ``center``, ``inverse``."""
    center = np.zeros(3, dtype=np.float32)
    centered = scan - center
    voxel_coords = np.floor(centered / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2
    z = voxel_centers[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
    data = {
        "coord": torch.from_numpy(voxel_centers).float(),
        "color": torch.from_numpy(colors).float(),
        "normal": torch.zeros_like(torch.from_numpy(voxel_centers)).float(),
        "grid_size": float(voxel_size),
        "batch": torch.zeros(voxel_centers.shape[0], dtype=torch.long),
    }
    return data, center, inverse


def build_scene_completion_model(args: argparse.Namespace, device: torch.device):
    """Same stack as ``training/train_diffusion.py`` (Sonata + diffusion)."""
    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=True,
        enable_flash=args.enable_flash,
        feature_levels=[0],
    )
    extractor = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    qrad = float(args.query_max_radius)
    if getattr(args, "scene_radius", 0) and float(args.scene_radius) > 0:
        qrad = min(qrad, float(args.scene_radius))
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=extractor,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        denoising_steps=args.denoising_steps,
        chamfer_max_points=args.chamfer_max_points,
        num_query_extra=getattr(args, "num_query_extra", None),
        query_radius=qrad,
        query_min_radius=float(getattr(args, "query_min_radius", QUERY_MIN_RADIUS)),
        conditioning_mode=str(getattr(args, "conditioning_mode", "concat")),
        conditioning_scale=float(getattr(args, "conditioning_scale", 1.0)),
    )
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def run_completion(
    model: SceneCompletionDiffusion,
    partial_scan: Dict[str, torch.Tensor],
    num_steps: int = 20,
    use_query_points: bool = True,
    use_amp: bool = False,
    **query_kwargs,
) -> np.ndarray:
    """Move tensors to model device; call ``model.complete_scene``; return numpy (centered)."""
    device = next(model.parameters()).device
    for k, v in partial_scan.items():
        if isinstance(v, torch.Tensor):
            partial_scan[k] = v.to(device)
    if device.type == "cuda" and use_amp:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.complete_scene(
                partial_scan,
                num_steps=num_steps,
                use_query_points=use_query_points,
                **query_kwargs,
            )
    else:
        out = model.complete_scene(
            partial_scan,
            num_steps=num_steps,
            use_query_points=use_query_points,
            **query_kwargs,
        )
    return out.float().cpu().numpy()


def main() -> None:
    args = parse_args()
    apply_completion_profile(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    t_wall0 = time.perf_counter()

    print(f"\nLoading scan: {args.input}")
    t0 = time.perf_counter()
    scan_full = load_scan(args.input)
    print(f"  {scan_full.shape[0]} points (file)")

    if args.scene_radius > 0:
        scan = crop_lidar_radius(scan_full, args.scene_radius)
        print(
            f"  Scene crop: Euclidean r ≤ {args.scene_radius} m → {scan.shape[0]} points "
            "(same frame as training: before voxelize)"
        )
        if scan.shape[0] == 0:
            raise ValueError(
                "No points left after --scene-radius crop; check radius or input frame."
            )
    else:
        scan = scan_full
    t1 = time.perf_counter()

    print("\nVoxelizing partial...")
    partial_data, center, _ = prepare_input(scan, voxel_size=args.voxel_size)
    n_partial = int(partial_data["coord"].shape[0])
    print(f"  {n_partial} voxels (size={args.voxel_size})")
    mp = int(getattr(args, "max_partial_points", 0) or 0)
    if mp > 0:
        before = n_partial
        partial_data = _subsample_partial_dict(partial_data, mp)
        n_partial = int(partial_data["coord"].shape[0])
        if n_partial < before:
            print(f"  --max-partial-points {mp}: subsampled {before} → {n_partial}")
    sp = getattr(args, "save_partial", None)
    if sp:
        pts_p = partial_data["coord"].float().cpu().numpy().astype(np.float64) + center.astype(
            np.float64
        )
        save_pointcloud(sp, pts_p.astype(np.float32))
        print(f"  Saved partial voxels → {sp}")
    t2 = time.perf_counter()

    use_query = not args.no_query_points
    k = int(args.num_query_extra)
    if k < 0:
        raise ValueError("--num-query-extra must be >= 0")
    print(
        f"--num-query-extra={k} → target_total_points={n_partial + k} "
        f"(K_query={k})"
    )
    t3 = time.perf_counter()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = build_scene_completion_model(args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t4 = time.perf_counter()

    qmax = float(args.query_max_radius)
    if args.scene_radius > 0:
        qmax = min(qmax, float(args.scene_radius))
    qmin = float(getattr(args, "query_min_radius", QUERY_MIN_RADIUS))
    if use_query:
        print(
            f"Denoising: extra queries in cylinder r_xy∈[{qmin}, {qmax}] m, "
            f"z∈[-R,R] (same as training; capped by scene-radius when set)"
        )
    else:
        print("Denoising: partial only")
    shell_coords_cpu: Optional[torch.Tensor] = None
    if use_query:
        shell_coords_cpu = build_shell_coords_single(
            partial_data["coord"].float(),
            num_query_extra=int(args.num_query_extra),
            target_shell_total=None,
            max_radius=qmax,
            min_radius=qmin,
            query_voxel_size=args.query_voxel_size,
            rng=None,
        )
    completion_profile = str(getattr(args, "completion_profile", "strict"))
    if completion_profile == "guided":
        print(
            f"Completion profile: guided (α={float(args.anchor_alpha):.3f}, "
            f"anchor_start_step={args.anchor_start_step})"
        )
    rc_kw: Dict[str, object] = dict(
        anchor_alpha=float(getattr(args, "anchor_alpha", 1.0)),
        anchor_start_step=getattr(args, "anchor_start_step", None),
        diagnostics=bool(getattr(args, "diagnostics", False)),
    )
    aas = rc_kw["anchor_start_step"]
    if aas is not None:
        print(
            f"Anchor: α={rc_kw['anchor_alpha']:.3f}, active for t_step ≤ {aas} "
            f"(of {args.denoising_steps} DDIM steps)"
        )
    elif float(rc_kw["anchor_alpha"]) < 1.0:
        print(f"Anchor: soft blend α={float(rc_kw['anchor_alpha']):.3f} on all steps")
    if shell_coords_cpu is not None:
        rc_kw["shell_coords"] = shell_coords_cpu
    snap_iters = _parse_denoise_snapshot_iters(
        getattr(args, "save_denoise_checkpoints", None)
    )
    if snap_iters:
        stem_out, _ = os.path.splitext(args.output)
        center_np = np.asarray(center, dtype=np.float64)

        def _on_denoise_snapshot(step: int, xt: torch.Tensor) -> None:
            path = f"{stem_out}_denoise_{int(step):03d}.ply"
            pts = (xt.float().cpu().numpy().astype(np.float64) + center_np).astype(
                np.float32
            )
            save_pointcloud(path, pts)

        rc_kw["denoise_snapshot_iters"] = snap_iters
        rc_kw["on_denoise_snapshot"] = _on_denoise_snapshot
        print(
            f"Denoise snapshots: steps {sorted(snap_iters)} → "
            f"{stem_out}_denoise_*.ply"
        )
    completed = run_completion(
        model,
        partial_data,
        num_steps=args.denoising_steps,
        use_query_points=use_query,
        use_amp=bool(getattr(args, "amp_inference", False)),
        **rc_kw,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t5 = time.perf_counter()

    completed = completed + center
    print(f"\nOutput: {completed.shape[0]} points → {args.output}")
    save_pointcloud(args.output, completed)
    t6 = time.perf_counter()

    print(
        "\n[timing] "
        f"scan_load+crop {t1 - t0:.2f}s | "
        f"voxelize {t2 - t1:.2f}s | "
        f"target_total {t3 - t2:.2f}s | "
        f"model_init+ckpt {t4 - t3:.2f}s | "
        f"complete_scene {t5 - t4:.2f}s | "
        f"save_ply {t6 - t5:.2f}s | "
        f"total {t6 - t_wall0:.2f}s"
    )

    if args.visualize:
        # Compare what the model saw (cropped) vs completion in the same frame.
        visualize_comparison(scan, completed)
    print("Done.")


# Backwards-compatible name for scripts that imported ``complete_scene``
complete_scene = run_completion


if __name__ == "__main__":
    main()
