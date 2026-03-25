"""
Inference for scene completion (Sonata-LiDiff).

Training matches targets on full-scene coordinates (GT voxel centers). If you only run
``complete_scene`` with the default query shell built from the *partial* scan, the denoiser
is evaluated at the wrong positions — outputs often look like unstructured noise.

Two supported inference targets (see also ``models.diffusion_module.SceneCompletionDiffusion.complete_scene``):

1. **Validation / debugging** — pass ground-truth map coordinates::
     --gt_npz path/to/XXXXXX.npz
   Points must live in the same LiDAR (sensor) frame as the input scan (as in training).

2. **Blind inference** — dense 3D grid in sensor frame::
     --target_grid -40 40 -40 40 -3 5 --grid_step 0.2
   Large grids are randomly subsampled to ``--max_target_points``.

``SemanticKITTI`` uses ``scan_center = 0``; partial and GT are kept in sensor coordinates.
We mirror that here (do **not** subtract the scan mean for centering).

Encoder input: ``grid_coord`` must match one row per ``coord`` voxel, shifted to be
non-negative (Sonata serialization); see ``prepare_input`` below.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion
from models.refinement_net import RefinementNetwork
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description="Scene completion: use --gt_npz or --target_grid so targets match training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True, help="Scan .bin / .pcd / .ply")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="output_completed.ply")
    p.add_argument("--denoising_steps", type=int, default=50)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.1, help="Partial voxel size (match training)")
    p.add_argument("--refinement_ckpt", type=str, default=None)
    p.add_argument("--up_factor", type=int, default=6)
    p.add_argument(
        "--gt_npz",
        type=str,
        default=None,
        help="GT map .npz (key 'points'); target_coords = GT in sensor frame (validation)",
    )
    p.add_argument(
        "--max_target_points",
        type=int,
        default=200_000,
        help="Max points for GT or dense grid (random subsample if larger)",
    )
    p.add_argument(
        "--target_grid",
        type=float,
        nargs=6,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        default=None,
        help="Dense grid bounds in sensor frame (m). Example: --target_grid -40 40 -40 40 -3 5",
    )
    p.add_argument("--grid_step", type=float, default=0.2, help="Grid spacing when using --target_grid")
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--enable_flash", action="store_true", help="Requires flash-attn in many Sonata builds")
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine", "sigmoid"])
    return p.parse_args()


def load_scan(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return scan[:, :3]
    if ext in (".pcd", ".ply"):
        if o3d is None:
            raise RuntimeError("open3d required for PCD/PLY")
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    raise ValueError(f"Unsupported file format: {ext}")


def save_pointcloud(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    points = np.asarray(points, dtype=np.float64)
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(path, pcd)
        print(f"Saved point cloud to {path}")
        return
    # ASCII PLY fallback
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = points.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points.astype(np.float64):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved point cloud to {path}")


def visualize_comparison(partial: np.ndarray, complete: np.ndarray) -> None:
    if o3d is None:
        print("open3d not installed; skip visualization")
        return
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
    scan: np.ndarray,
    voxel_size: float = 0.1,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Voxelize in LiDAR frame (origin = sensor). Matches ``SemanticKITTI`` / ``inference.py``."""
    center = np.zeros(3, dtype=np.float32)
    centered = scan.astype(np.float32) - center
    voxel_coords = np.floor(centered / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_centers = unique_voxels.astype(np.float32) * voxel_size + voxel_size / 2.0
    grid_coord = (unique_voxels - unique_voxels.min(axis=0)).astype(np.int64)

    z = voxel_centers[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, 1.0 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
    data: Dict[str, torch.Tensor] = {
        "coord": torch.from_numpy(voxel_centers).float(),
        "color": torch.from_numpy(colors).float(),
        "normal": torch.zeros_like(torch.from_numpy(voxel_centers)).float(),
        "grid_coord": torch.from_numpy(grid_coord).long(),
        "grid_size": float(voxel_size),
        "batch": torch.zeros(voxel_centers.shape[0], dtype=torch.long),
    }
    return data, center, inverse


def _subsample_points(pts: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    n = pts.shape[0]
    if n <= max_n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_n, replace=False)
    return pts[idx]


def build_target_coords(
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if args.gt_npz:
        data = np.load(args.gt_npz)
        gt = np.asarray(data["points"], dtype=np.float32)
        gt = _subsample_points(gt, int(args.max_target_points))
        return torch.from_numpy(gt.copy()).float().to(device)
    if args.target_grid is not None:
        if len(args.target_grid) != 6:
            raise ValueError("--target_grid needs 6 values: x0 x1 y0 y1 z0 z1")
        x0, x1, y0, y1, z0, z1 = [float(x) for x in args.target_grid]
        step = float(args.grid_step)
        xs = np.arange(x0, x1 + 1e-9, step, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-9, step, dtype=np.float64)
        zs = np.arange(z0, z1 + 1e-9, step, dtype=np.float64)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
        grid = _subsample_points(grid, int(args.max_target_points))
        return torch.from_numpy(grid).float().to(device)
    return None


@torch.no_grad()
def run_completion(
    model: SceneCompletionDiffusion,
    partial_scan: Dict[str, torch.Tensor],
    num_steps: int,
    target_coords: Optional[torch.Tensor] = None,
) -> np.ndarray:
    device = next(model.parameters()).device
    for key, val in partial_scan.items():
        if isinstance(val, torch.Tensor):
            partial_scan[key] = val.to(device)
    kw: Dict[str, object] = {"num_steps": num_steps}
    if target_coords is not None:
        kw["target_coords"] = target_coords
        kw["use_query_points"] = False
    out = model.complete_scene(partial_scan, **kw)
    return out.float().cpu().numpy()


def main():
    args = parse_args()

    if args.gt_npz is None and args.target_grid is None:
        print(
            "Warning: neither --gt_npz nor --target_grid set. "
            "Denoising will use only the partial-voxel + synthetic shell layout; "
            "this often mismatches training (full GT coordinates). "
            "Prefer --gt_npz for sanity checks or --target_grid for blind inference.\n"
        )

    print(f"\nLoading scan: {args.input}")
    scan = load_scan(args.input)
    print(f"  {scan.shape[0]} points")

    partial_data, center, _ = prepare_input(scan, voxel_size=args.voxel_size)
    print(f"  Partial voxels: {partial_data['coord'].shape[0]} (voxel_size={args.voxel_size})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_coords = build_target_coords(args, device)
    if target_coords is not None:
        print(f"  target_coords: {target_coords.shape[0]} points")

    print(f"\nLoading model: {args.checkpoint}")
    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=True,
        enable_flash=args.enable_flash,
        feature_levels=[2, 3, 4],
    )
    condition_extractor = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[2, 3, 4],
        fusion_type="attention",
    )
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        denoising_steps=args.denoising_steps,
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("\nCompleting scene...")
    completed = run_completion(
        model, partial_data, num_steps=args.denoising_steps, target_coords=target_coords
    )

    if args.refinement_ckpt and os.path.exists(args.refinement_ckpt):
        print("Applying refinement...")
        refinement = RefinementNetwork(up_factor=args.up_factor).to(device)
        rckpt = load_checkpoint(args.refinement_ckpt)
        refinement.load_state_dict(rckpt.get("model_state_dict", rckpt), strict=True)
        refinement.eval()
        with torch.no_grad():
            completed = refinement(torch.from_numpy(completed).float().to(device)).cpu().numpy()

    completed = completed + center.astype(np.float64)
    print(f"Output: {completed.shape[0]} points → {args.output}")
    save_pointcloud(args.output, completed.astype(np.float32))

    if args.visualize:
        visualize_comparison(scan, completed)
    print("Done.")


if __name__ == "__main__":
    main()
