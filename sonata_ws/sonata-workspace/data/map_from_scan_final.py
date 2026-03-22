#!/usr/bin/env python3
"""
Ground truth map generation for SemanticKITTI — final consolidated version.

Starts from map_from_scans.py and adds proven features:
  - Crop in world around fused cloud (center=mean, output_radius) before voxel
  - Optional ICP merge (from new_map_from_scans)
  - Optional refine: SOR + ROR (scipy fallback, no Open3D display)
  - CLI: scan_ids, force, output_subdir, quiet, window, output_radius

Pipeline: load window → merge (pose or ICP) → crop in world → voxel → transform to scan → refine? → subsample → NPZ

  python data/map_from_scan_final.py -p .../sequences -s 00 --scan_ids 000000 --refine
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DATA_DIR)
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from map_from_scans import load_poses, voxelize  # noqa: E402

_DEFAULT_GT_EXPORT = os.path.join(_REPO_ROOT, "gt_maps_refined")


def _crop_radius(pts: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Points within radius of center (inclusive)."""
    if radius <= 0 or pts.shape[0] == 0:
        return pts
    d = np.linalg.norm(pts[:, :3] - np.asarray(center).reshape(1, 3), axis=1)
    return pts[d <= radius]


def _load_scan_world(
    velodyne_dir: str,
    labels_dir: str,
    scan_files: list,
    poses: list,
    idx: int,
    *,
    ego_min_range: float = 3.5,
    presample_cap: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load scan, filter moving objects and near-origin, subsample, transform to world."""
    sf = scan_files[idx]
    sp = os.path.join(velodyne_dir, sf)
    pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
    lp = os.path.join(labels_dir, sf.replace(".bin", ".label"))
    if os.path.exists(lp):
        lb = np.fromfile(lp, dtype=np.uint32) & 0xFFFF
        mask = (lb < 252) | (lb > 259)
        if mask.sum() < len(lb):
            pts = pts[mask]
    dist = np.linalg.norm(pts[:, :3], axis=1)
    pts = pts[dist > ego_min_range]
    if len(pts) > presample_cap:
        r = rng if rng is not None else np.random.default_rng(42)
        idx_sub = r.choice(len(pts), presample_cap, replace=False)
        pts = pts[idx_sub]
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts[:, :3], ones])
    return (poses[idx] @ homo.T).T[:, :3]


def _refine_scipy(
    points: np.ndarray,
    stat_nb: int = 20,
    stat_std: float = 1.0,
    radius_nb: int = 5,
    radius: float = 0.3,
    floater_radius: float = 0.0,
    floater_min: int = 10,
) -> np.ndarray:
    """SOR + ROR via scipy; optional floater pass."""
    from scipy.spatial import cKDTree

    if points.shape[0] == 0:
        return points.astype(np.float32)
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    # Statistical
    if n > 2:
        k = min(stat_nb, n - 1)
        if k >= 1:
            tree = cKDTree(pts)
            try:
                dists, _ = tree.query(pts, k=k + 1, workers=-1)
            except TypeError:
                dists, _ = tree.query(pts, k=k + 1)
            mean_d = dists[:, 1:].mean(axis=1)
            mu, sigma = float(mean_d.mean()), float(mean_d.std()) + 1e-9
            pts = pts[mean_d <= mu + stat_std * sigma]
            n = pts.shape[0]

    # Radius
    if n > 0:
        tree = cKDTree(pts)
        try:
            neighbors = tree.query_ball_point(pts, r=radius, workers=-1)
        except TypeError:
            neighbors = tree.query_ball_point(pts, r=radius)
        counts = np.array([len(neighbors[i]) for i in range(len(neighbors))])
        pts = pts[counts >= radius_nb]
        n = pts.shape[0]

    # Floater
    if floater_radius > 0 and n > 0:
        tree = cKDTree(pts)
        try:
            neighbors = tree.query_ball_point(pts, r=floater_radius, workers=-1)
        except TypeError:
            neighbors = tree.query_ball_point(pts, r=floater_radius)
        counts = np.array([len(neighbors[i]) for i in range(len(neighbors))])
        pts = pts[counts >= floater_min]

    return pts.astype(np.float32)


def generate_sequence_map(
    seq_path: str,
    output_dir: str,
    *,
    voxel_size: float = 0.1,
    backend: str = "open3d",
    sequences: list | None = None,
    output_subdir: str = "ground_truth",
    scan_ids_filter: set | None = None,
    force: bool = False,
    quiet: bool = False,
    window: int = 25,
    max_gt_points: int = 200_000,
    output_radius: float = 20.0,
    use_icp: bool = False,
    use_refine: bool = False,
    refine_floaters: bool = False,
) -> None:
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    rng = np.random.default_rng(42)

    for seq in sequences:
        seq_folder = os.path.join(seq_path, seq)
        velodyne_dir = os.path.join(seq_folder, "velodyne")
        labels_dir = os.path.join(seq_folder, "labels")
        poses_path = os.path.join(seq_folder, "poses.txt")
        calib_path = os.path.join(seq_folder, "calib.txt")

        if not os.path.exists(velodyne_dir):
            print(f"Skipping {seq}: velodyne not found")
            continue
        if not os.path.exists(poses_path):
            print(f"Skipping {seq}: poses.txt not found")
            continue

        scan_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith(".bin")])
        if len(scan_files) == 0:
            print(f"Skipping {seq}: no .bin files")
            continue

        poses = load_poses(calib_path, poses_path, show_progress=not quiet)
        if len(poses) != len(scan_files):
            print(f"Warning {seq}: {len(poses)} poses vs {len(scan_files)} scans")

        n_scans = len(scan_files)
        indices = list(range(n_scans))
        if scan_ids_filter is not None:
            indices = [
                i
                for i in range(n_scans)
                if scan_files[i].replace(".bin", "") in scan_ids_filter
            ]
            if len(indices) == 0:
                print(f"  No scan_ids from {scan_ids_filter} in {seq}, skip")
                continue

        gt_seq_dir = os.path.join(output_dir, output_subdir, seq) if output_subdir else os.path.join(output_dir, seq)
        os.makedirs(gt_seq_dir, exist_ok=True)

        def load_scan(idx: int) -> np.ndarray:
            return _load_scan_world(
                velodyne_dir, labels_dir, scan_files, poses, idx,
                ego_min_range=3.5, presample_cap=50_000, rng=rng,
            )

        if scan_ids_filter is not None:
            needed = set()
            for i in indices:
                lo, hi = max(0, i - window), min(n_scans, i + window + 1)
                needed.update(range(lo, hi))
            to_load = sorted(needed)
            if not quiet:
                print(f"  Loading {len(to_load)} scans (window {window}) for {seq}...")
        else:
            to_load = list(range(n_scans))
            if not quiet:
                print(f"  Loading {n_scans} scans for {seq}...")

        scan_cache = {}
        for idx in tqdm(to_load, desc=f"Load {seq}", leave=False, disable=quiet):
            scan_cache[idx] = load_scan(idx)

        merge_fn = None
        if use_icp:
            try:
                from new_map_from_scans import merge_window_points_icp
                merge_fn = merge_window_points_icp
            except (ImportError, OSError) as e:
                if not quiet:
                    print(f"  ICP unavailable ({e}), using pose-only merge")

        for i in tqdm(indices, desc=f"Seq {seq}", disable=quiet):
            scan_id = scan_files[i].replace(".bin", "")
            out_path = os.path.join(gt_seq_dir, f"{scan_id}.npz")
            if os.path.exists(out_path) and not force:
                continue

            lo = max(0, i - window)
            hi = min(n_scans, i + window + 1)
            local_pts = []
            for j in range(lo, hi):
                pts = scan_cache[j]
                center_j = poses[j][:3, 3]
                dist = np.linalg.norm(pts[:, :3] - center_j, axis=1)
                pts = pts[dist < 15.0]
                local_pts.append(pts)

            if merge_fn is not None:
                mid_pts = local_pts[len(local_pts) // 2]
                center = np.mean(mid_pts[:, :3], axis=0) if len(mid_pts) > 0 else poses[i][:3, 3]
                local_pts = [_crop_radius(p, center, 30.0) for p in local_pts]
                all_pts = merge_fn(
                    local_pts,
                    icp_voxel=0.25,
                    icp_threshold=1.0,
                    icp_max_iter=30,
                    icp_target_history=8,
                    icp_scales=[0.5, 0.2, 0.08],
                    icp_point_to_plane=True,
                    icp_robust=False,
                    icp_normals_finest_only=True,
                    icp_rng_seed=42,
                )
            else:
                all_pts = np.vstack(local_pts).astype(np.float32)

            # Crop in world around ego (stable reference); fallback if empty
            if output_radius > 0 and all_pts.shape[0] > 0:
                ego = poses[i][:3, 3]
                cropped = _crop_radius(all_pts, ego, output_radius)
                if cropped.shape[0] > 0:
                    all_pts = cropped

            if all_pts.shape[0] == 0:
                continue

            # Refine before voxelization (preserves structure, reduces holes)
            if use_refine:
                all_pts = _refine_scipy(
                    all_pts,
                    stat_nb=20, stat_std=1.0,
                    radius_nb=5, radius=0.3,
                    floater_radius=0.5 if refine_floaters else 0.0,
                    floater_min=10,
                )
                if all_pts.shape[0] == 0:
                    continue

            # Dense core voxelization: finer near ego, coarser far
            ego = poses[i][:3, 3]
            dist = np.linalg.norm(all_pts[:, :3] - ego, axis=1)
            near = all_pts[dist < 8.0]
            far = all_pts[dist >= 8.0]
            voxels = []
            if len(near) > 0:
                voxels.append(voxelize(near, 0.05, backend=backend))
            if len(far) > 0:
                voxels.append(voxelize(far, voxel_size, backend=backend))
            if len(voxels) == 0:
                continue
            map_voxel = np.vstack(voxels)
            del all_pts

            pose_inv = np.linalg.inv(poses[i])
            ones = np.ones((map_voxel.shape[0], 1))
            map_scan = (pose_inv @ np.hstack([map_voxel, ones]).T).T[:, :3]
            del map_voxel

            if len(map_scan) > max_gt_points:
                idx_sub = rng.choice(len(map_scan), max_gt_points, replace=False)
                map_scan = map_scan[idx_sub]

            np.savez_compressed(out_path, points=map_scan.astype(np.float32))

        del scan_cache
        rel = os.path.join(output_subdir, seq) if output_subdir else seq
        if not quiet:
            print(f"Saved {seq} -> {rel}/")


def main():
    _default_root = os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset")
    _default_sequences = os.path.join(_default_root, "sequences")
    parser = argparse.ArgumentParser(
        description="GT maps from SemanticKITTI — final pipeline (crop, ICP, refine)"
    )
    parser.add_argument("-p", "--path", type=str, default=_default_sequences)
    parser.add_argument("-o", "--output", type=str, default=_DEFAULT_GT_EXPORT)
    parser.add_argument("-s", "--sequences", type=str, nargs="+", default=None)
    parser.add_argument("--output_subdir", type=str, default="ground_truth")
    parser.add_argument("--scan_ids", type=str, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=float, default=0.1)
    parser.add_argument("-b", "--backend", choices=["numpy", "open3d", "torch", "auto"], default="open3d")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--max_gt_points", type=int, default=200_000)
    parser.add_argument("--output_radius", type=float, default=20.0,
        help="Crop in world (m) around fused cloud before voxel; 0=no crop")
    parser.add_argument("--icp", action="store_true", help="ICP merge in window (Open3D)")
    parser.add_argument("--refine", action="store_true",
        help="SOR+ROR outlier removal (scipy, no Open3D display)")
    parser.add_argument("--refine_floaters", action="store_true",
        help="Extra floater pass after refine")

    args = parser.parse_args()
    seq_path = args.path.rstrip("/")
    output_dir = os.path.abspath(args.output.rstrip("/"))
    sequences = args.sequences or ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    scan_filter = set(args.scan_ids) if args.scan_ids else None

    print(f"Sequences: {seq_path}")
    print(f"Output: {output_dir}/{args.output_subdir}")
    print(f"Window={args.window}, output_radius={args.output_radius}, refine={args.refine}, icp={args.icp}")

    generate_sequence_map(
        seq_path=seq_path,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        backend=args.backend,
        sequences=sequences,
        output_subdir=args.output_subdir,
        scan_ids_filter=scan_filter,
        force=args.force,
        quiet=args.quiet,
        window=args.window,
        max_gt_points=args.max_gt_points,
        output_radius=args.output_radius,
        use_icp=args.icp,
        use_refine=args.refine,
        refine_floaters=args.refine_floaters,
    )
    print("Done.")


if __name__ == "__main__":
    main()
