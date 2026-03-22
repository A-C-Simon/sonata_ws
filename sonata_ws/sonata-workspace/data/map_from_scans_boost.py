#!/usr/bin/env python3
"""
Baseline LiDAR map fusion (boost): sliding window in world, voxelize, transform to scan frame.

Pipeline: window crop → merge (pose-only + optional ICP) → radius crop in world (ego) →
voxelize → SOR/ROR → transform to scan frame → subsample → NPZ.

Output: ground_truth/{seq}/{scan_id}.npz with key 'points'

  python data/map_from_scans_boost.py -p .../sequences -s 00 --scan_ids 000000
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DATA_DIR)
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from map_from_scans import load_poses, voxelize  # noqa: E402

_DEFAULT_GT_EXPORT = os.path.join(_REPO_ROOT, "gt_maps_refined")


@dataclass(frozen=True)
class BoostDefaults:
    """Defaults for pose-only boost pipeline (CLI + API)."""

    voxel_size: float = 0.1
    backend: str = "open3d"
    output_subdir: str = "ground_truth"
    max_gt_points: int = 200_000
    window_half: int = 20
    accumulation_radius: float = 15.0
    output_radius: float = 20.0
    force: bool = False
    quiet: bool = False
    output_name_suffix: str = ""
    scan_ego_min_range_m: float = 3.5
    scan_load_presample_cap: int = 50_000
    # SOR / ROR / ICP refinement
    use_sor: bool = True
    use_ror: bool = True
    sor_nb_neighbors: int = 12
    sor_std_ratio: float = 2.0
    ror_nb_points: int = 5
    ror_radius: float = 0.5
    use_icp: bool = True
    icp_max_iter: int = 5
    icp_threshold: float = 1.0
    icp_downsample: float = 0.25  # voxel size before ICP (~5-10x speedup)


BOOST = BoostDefaults()

BOOST_DEFAULT_SEQUENCES: tuple[str, ...] = (
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
)


def sor_filter(
    points: np.ndarray,
    nb_neighbors: int = 12,
    std_ratio: float = 2.0,
    max_points: int = 100_000,
) -> np.ndarray:
    """Statistical Outlier Removal. Downsample if >max_points for speed."""
    if points.shape[0] < 50:
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, :3], dtype=np.float64))
    if points.shape[0] > max_points:
        pcd = pcd.voxel_down_sample(0.12)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return np.asarray(pcd.points, dtype=np.float64)


def ror_filter(
    points: np.ndarray,
    nb_points: int = 5,
    radius: float = 0.5,
    max_points: int = 100_000,
) -> np.ndarray:
    """Radius Outlier Removal. Downsample if >max_points for speed."""
    if points.shape[0] < 50:
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, :3], dtype=np.float64))
    if points.shape[0] > max_points:
        pcd = pcd.voxel_down_sample(0.12)
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    return np.asarray(pcd.points, dtype=np.float64)


def _valid_icp(T: np.ndarray, max_t: float = 0.5, max_deg: float = 5.0) -> bool:
    """Reject ICP transform if translation or rotation too large (prevents drift)."""
    t = np.linalg.norm(T[:3, 3])
    R = T[:3, :3]
    val = (np.trace(R) - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    ang = np.degrees(np.arccos(val))
    return t < max_t and ang < max_deg


def fast_icp_align(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    max_iter: int = 10,
    threshold: float = 1.0,
    downsample_voxel: float = 0.0,
) -> np.ndarray:
    """Lightweight point-to-point ICP. Downsample before ICP for speed; T applied to full cloud."""
    if src_pts.shape[0] < 50 or tgt_pts.shape[0] < 50:
        return src_pts.copy()
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(src_pts[:, :3], dtype=np.float64))
    tgt.points = o3d.utility.Vector3dVector(np.asarray(tgt_pts[:, :3], dtype=np.float64))
    if downsample_voxel > 0:
        src = src.voxel_down_sample(downsample_voxel)
        tgt = tgt.voxel_down_sample(downsample_voxel)
        if len(src.points) < 20 or len(tgt.points) < 20:
            src = o3d.geometry.PointCloud()
            tgt = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(np.asarray(src_pts[:, :3], dtype=np.float64))
            tgt.points = o3d.utility.Vector3dVector(np.asarray(tgt_pts[:, :3], dtype=np.float64))
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    T = np.asarray(reg.transformation, dtype=np.float64)
    if not _valid_icp(T):
        return np.asarray(src_pts[:, :3], dtype=np.float64).copy()
    pts_h = np.hstack([np.asarray(src_pts[:, :3], dtype=np.float64), np.ones((src_pts.shape[0], 1))])
    aligned_full = (T @ pts_h.T).T[:, :3]
    return aligned_full


def boost_finalize_frame_from_fused(
    i: int,
    all_pts: np.ndarray,
    local_pts: list,
    c: dict,
) -> None:
    """Crop ``output_radius`` around ego in world, voxelize, then ``inv(poses[i])`` to scan ``i``."""
    del local_pts
    scan_files = c["scan_files"]
    poses = c["poses"]
    gt_seq_dir = c["gt_seq_dir"]
    output_name_suffix = c["output_name_suffix"]
    force = c["force"]
    voxel_size = c["voxel_size"]
    backend = c["backend"]

    scan_id = scan_files[i].replace(".bin", "")
    out_path = _boost_gt_npz_path(gt_seq_dir, scan_files, i, output_name_suffix)
    if os.path.exists(out_path) and not force:
        return

    ego_w = np.asarray(poses[i][:3, 3], dtype=np.float64)
    pose_inv = np.linalg.inv(np.asarray(poses[i], dtype=np.float64))

    output_radius = float(c["output_radius"])
    if output_radius > 0 and all_pts.shape[0] > 0:
        all_pts_f64 = np.asarray(all_pts, dtype=np.float64)
        n_before = all_pts_f64.shape[0]
        cropped = crop_by_radius(all_pts_f64, ego_w, output_radius)
        if cropped.shape[0] == 0:
            print(f"WARNING: empty crop at frame {i}")
        elif n_before > 500 and cropped.shape[0] < n_before * 0.1:
            print(
                f"WARNING: crop removed >90% points at frame {i} "
                f"({n_before} -> {cropped.shape[0]})"
            )
        all_pts = cropped

    if all_pts.shape[0] < 100:
        print(
            f"WARNING: low point count after fusion at frame {i}: {all_pts.shape[0]}"
        )

    map_voxel = voxelize(all_pts, voxel_size, backend=backend)
    if c.get("use_sor", False):
        map_voxel = sor_filter(
            map_voxel,
            nb_neighbors=c.get("sor_nb_neighbors", 20),
            std_ratio=c.get("sor_std_ratio", 2.0),
        )
    if c.get("use_ror", False):
        map_voxel = ror_filter(
            map_voxel,
            nb_points=c.get("ror_nb_points", 5),
            radius=c.get("ror_radius", 0.5),
        )

    ones = np.ones((map_voxel.shape[0], 1))
    map_scan = (
        pose_inv @ np.hstack([np.asarray(map_voxel, dtype=np.float64), ones]).T
    ).T[:, :3].astype(np.float32)

    max_gt_points = c["max_gt_points"]
    if len(map_scan) > max_gt_points:
        idx_sub = np.random.choice(len(map_scan), max_gt_points, replace=False)
        map_scan = map_scan[idx_sub]

    np.savez_compressed(out_path, points=map_scan.astype(np.float32))


def crop_by_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Keep points strictly inside a ball (dist < radius)."""
    if radius is None or radius <= 0 or points.shape[0] == 0:
        return points
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    dist = np.linalg.norm(points[:, :3] - c, axis=1)
    return points[dist < radius]


def crop_window_scan_for_merge(
    pts: np.ndarray,
    scan_idx: int,
    poses: np.ndarray,
    radius: float,
) -> np.ndarray:
    """World crop: ball at scan ego pose."""
    if radius is None or radius <= 0 or pts.shape[0] == 0:
        return pts
    c = poses[scan_idx][:3, 3].astype(np.float64)
    return crop_by_radius(pts, c, radius)


def symmetric_window_bounds(i: int, n_scans: int, half: int) -> tuple[int, int]:
    """Return [lo, hi) for frames i-half … i+half inclusive."""
    if n_scans <= 0 or half < 0:
        return 0, 0
    lo = max(0, i - half)
    hi = min(n_scans, i + half + 1)
    return lo, hi


def _boost_gt_npz_path(
    gt_seq_dir: str, scan_files: list, frame_idx: int, name_suffix: str
) -> str:
    sid = scan_files[frame_idx].replace(".bin", "")
    base = f"{sid}{name_suffix}" if name_suffix else sid
    return os.path.join(gt_seq_dir, f"{base}.npz")


def _load_kitti_scan_to_world(
    idx: int,
    scan_files: list,
    velodyne_dir: str,
    labels_dir: str,
    poses: np.ndarray,
) -> np.ndarray:
    """One SemanticKITTI .bin → world XYZ (pose applied), ego-range filter + cap."""
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
    pts = pts[dist > BOOST.scan_ego_min_range_m]
    cap = BOOST.scan_load_presample_cap
    if len(pts) > cap:
        pts = pts[np.random.choice(len(pts), cap, replace=False)]
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts[:, :3], ones])
    return (poses[idx] @ homo.T).T[:, :3]


class SlidingMapFusion:
    """Sliding window with pose-aligned scans in world. Optional lightweight ICP refinement."""

    def __init__(
        self,
        window_half: int,
        n_scans: int,
        poses: np.ndarray,
        accumulation_radius: float,
        use_icp: bool = True,
        icp_max_iter: int = 10,
        icp_threshold: float = 1.0,
        icp_downsample: float = 0.25,
    ) -> None:
        self.window_half = window_half
        self.n_scans = n_scans
        self.poses = poses
        self.accumulation_radius = accumulation_radius
        self.use_icp = use_icp
        self.icp_max_iter = icp_max_iter
        self.icp_threshold = icp_threshold
        self.icp_downsample = icp_downsample
        self._scan_cache: dict | None = None
        self.active_indices: list[int] = []
        self.active_scans: list[np.ndarray] = []
        self._output_idx: int = 0

    def _crop_scan(self, j: int) -> np.ndarray:
        pts_j = self._scan_cache[j]
        return crop_window_scan_for_merge(
            pts_j, j, self.poses, self.accumulation_radius
        )

    def window_cropped_raw(self) -> list:
        return [self._crop_scan(j) for j in self.active_indices]

    def initialize(self, center_idx: int, scan_cache: dict) -> np.ndarray:
        self._scan_cache = scan_cache
        self._output_idx = center_idx
        lo, hi = symmetric_window_bounds(center_idx, self.n_scans, self.window_half)
        aligned_list: list[np.ndarray] = []
        for j in range(lo, hi):
            raw = self._crop_scan(j)
            raw64 = np.asarray(raw, dtype=np.float64)
            if self.use_icp and len(aligned_list) > 0:
                aligned = fast_icp_align(
                    raw64,
                    aligned_list[-1],
                    max_iter=self.icp_max_iter,
                    threshold=self.icp_threshold,
                    downsample_voxel=self.icp_downsample,
                )
            else:
                aligned = raw64
            aligned_list.append(aligned)
        self.active_indices = list(range(lo, hi))
        self.active_scans = aligned_list
        return np.vstack(self.active_scans).astype(np.float32)

    def update(self, new_center_idx: int, scan_cache: dict) -> np.ndarray:
        self._scan_cache = scan_cache
        self._output_idx = new_center_idx
        new_lo, new_hi = symmetric_window_bounds(
            new_center_idx, self.n_scans, self.window_half
        )
        while self.active_indices and self.active_indices[0] < new_lo:
            self.active_indices.pop(0)
            self.active_scans.pop(0)

        j_next = new_lo if not self.active_indices else self.active_indices[-1] + 1
        while j_next < new_hi:
            raw = self._crop_scan(j_next)
            raw64 = np.asarray(raw, dtype=np.float64)
            run_icp = self.use_icp and len(self.active_scans) > 0
            if run_icp:
                center = self.poses[j_next][:3, 3]
                mask_src = np.linalg.norm(raw64[:, :3] - center, axis=1) < 8.0
                mask_tgt = np.linalg.norm(
                    self.active_scans[-1][:, :3] - center, axis=1
                ) < 8.0
                if mask_src.sum() > 50 and mask_tgt.sum() > 50:
                    aligned = fast_icp_align(
                        raw64,
                        self.active_scans[-1],
                        max_iter=self.icp_max_iter,
                        threshold=self.icp_threshold,
                        downsample_voxel=self.icp_downsample,
                    )
                else:
                    aligned = raw64
            else:
                aligned = raw64
            self.active_indices.append(j_next)
            self.active_scans.append(aligned)
            j_next += 1

        if not self.active_scans:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(self.active_scans).astype(np.float32)


def generate_sequence_map_boost(
    seq_path: str,
    output_dir: str,
    voxel_size: float = BOOST.voxel_size,
    sequences: list = None,
    backend: str = BOOST.backend,
    output_subdir: str = BOOST.output_subdir,
    scan_ids_filter: set = None,
    output_name_suffix: str = BOOST.output_name_suffix,
    force: bool = BOOST.force,
    window_half: int = BOOST.window_half,
    max_gt_points: int = BOOST.max_gt_points,
    accumulation_radius: float = BOOST.accumulation_radius,
    output_radius: float = BOOST.output_radius,
    quiet: bool = BOOST.quiet,
    use_sor: bool = BOOST.use_sor,
    use_ror: bool = BOOST.use_ror,
    sor_nb_neighbors: int = BOOST.sor_nb_neighbors,
    sor_std_ratio: float = BOOST.sor_std_ratio,
    ror_nb_points: int = BOOST.ror_nb_points,
    ror_radius: float = BOOST.ror_radius,
    use_icp: bool = BOOST.use_icp,
    icp_max_iter: int = BOOST.icp_max_iter,
    icp_threshold: float = BOOST.icp_threshold,
    icp_downsample: float = BOOST.icp_downsample,
    **kwargs,
) -> None:
    _ = kwargs

    if sequences is None:
        sequences = list(BOOST_DEFAULT_SEQUENCES)

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
                print(f"  No scan_ids from {scan_ids_filter} found in {seq}, skip")
                continue
            print(f"  Filter: only scan_ids {scan_ids_filter} -> {len(indices)} frames")

        if output_subdir:
            gt_seq_dir = os.path.join(output_dir, output_subdir, seq)
        else:
            gt_seq_dir = os.path.join(output_dir, seq)
        os.makedirs(gt_seq_dir, exist_ok=True)

        if scan_ids_filter is not None:
            needed = set()
            for i in indices:
                lo, hi = symmetric_window_bounds(i, n_scans, window_half)
                needed.update(range(lo, hi))
            to_load = sorted(needed)
            print(
                f"  Loading {len(to_load)} scans "
                f"(±{window_half} around index) for boost GT..."
            )
        else:
            to_load = list(range(n_scans))
            print(f"  Loading {n_scans} scans...")

        scan_cache = {}
        for idx in tqdm(to_load, desc=f"Loading {seq}", leave=False, disable=quiet):
            scan_cache[idx] = _load_kitti_scan_to_world(
                idx, scan_files, velodyne_dir, labels_dir, poses
            )

        worker_ctx = {
            "scan_files": scan_files,
            "poses": poses,
            "gt_seq_dir": gt_seq_dir,
            "output_name_suffix": output_name_suffix,
            "force": force,
            "window_half": window_half,
            "n_scans": n_scans,
            "accumulation_radius": accumulation_radius,
            "voxel_size": voxel_size,
            "backend": backend,
            "max_gt_points": max_gt_points,
            "output_radius": output_radius,
            "quiet": quiet,
            "use_sor": use_sor,
            "use_ror": use_ror,
            "sor_nb_neighbors": sor_nb_neighbors,
            "sor_std_ratio": sor_std_ratio,
            "ror_nb_points": ror_nb_points,
            "ror_radius": ror_radius,
        }

        if not quiet:
            mode = "ICP refinement" if use_icp else "pose-only"
            print(f"  Incremental sliding window ({mode}): reuse scans between consecutive frames")

        fusion = SlidingMapFusion(
            window_half,
            n_scans,
            poses,
            accumulation_radius,
            use_icp=use_icp,
            icp_max_iter=icp_max_iter,
            icp_threshold=icp_threshold,
            icp_downsample=icp_downsample,
        )
        prev_processed: int | None = None
        for i in tqdm(indices, desc=f"Sequence {seq} (boost)", disable=quiet):
            out_path = _boost_gt_npz_path(
                gt_seq_dir, scan_files, i, output_name_suffix
            )
            if os.path.exists(out_path) and not force:
                continue
            if prev_processed is not None and i == prev_processed + 1:
                all_pts = fusion.update(i, scan_cache)
            else:
                all_pts = fusion.initialize(i, scan_cache)
            local_pts = fusion.window_cropped_raw()
            boost_finalize_frame_from_fused(i, all_pts, local_pts, worker_ctx)
            prev_processed = i

        del scan_cache
        rel = os.path.join(output_subdir, seq) if output_subdir else seq
        print(f"Saved boost ground truth for sequence {seq} -> {rel}/")


def main():
    _default_dataset = os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset")
    _default_sequences = os.path.join(_default_dataset, "sequences")
    parser = argparse.ArgumentParser(
        description="Baseline LiDAR GT: pose-only window merge → voxel → scan frame → NPZ"
    )
    parser.add_argument("-p", "--path", type=str, default=_default_sequences)
    parser.add_argument("-o", "--output", type=str, default=_DEFAULT_GT_EXPORT)
    parser.add_argument("--voxel_size", "-v", type=float, default=BOOST.voxel_size)
    parser.add_argument("--sequences", "-s", type=str, nargs="+", default=None)
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["numpy", "open3d", "torch", "auto", "auto_cuda"],
        default=BOOST.backend,
    )
    parser.add_argument("--gpu_voxelize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_subdir", type=str, default=BOOST.output_subdir)
    parser.add_argument("--scan_ids", nargs="+", default=None)
    parser.add_argument("--name_suffix", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max_gt_points",
        type=int,
        default=BOOST.max_gt_points,
        help="Random cap on points per saved frame after voxel + crop",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=BOOST.window_half,
        help="Half-window ±N scans (default 20 → 41 scans)",
    )
    parser.add_argument(
        "--accumulation_radius",
        "--crop_radius",
        type=float,
        default=BOOST.accumulation_radius,
        dest="accumulation_radius",
    )
    parser.add_argument("--output_radius", type=float, default=BOOST.output_radius)
    parser.add_argument("--no-sor", action="store_true", dest="no_sor", help="Disable SOR")
    parser.add_argument("--no-ror", action="store_true", dest="no_ror", help="Disable ROR")
    parser.add_argument("--no-icp", action="store_true", dest="no_icp", help="Disable ICP refinement")
    parser.add_argument("--sor-neighbors", type=int, default=BOOST.sor_nb_neighbors)
    parser.add_argument("--sor-std-ratio", type=float, default=BOOST.sor_std_ratio)
    parser.add_argument("--ror-nb-points", type=int, default=BOOST.ror_nb_points)
    parser.add_argument("--ror-radius", type=float, default=BOOST.ror_radius)
    parser.add_argument("--icp-max-iter", type=int, default=BOOST.icp_max_iter)
    parser.add_argument("--icp-threshold", type=float, default=BOOST.icp_threshold)
    parser.add_argument(
        "--icp-downsample",
        type=float,
        default=BOOST.icp_downsample,
        help="Voxel size for ICP pre-downsample (0=off, 0.25=fast)",
    )

    args = parser.parse_args()

    if args.gpu_voxelize:
        try:
            import torch
            if torch.cuda.is_available():
                args.backend = "torch"
            elif not args.quiet:
                print("  Note: --gpu_voxelize ignored (no CUDA)")
        except ImportError:
            if not args.quiet:
                print("  Note: --gpu_voxelize ignored (PyTorch not installed)")

    seq_path = args.path.rstrip("/")
    output_dir = os.path.abspath(args.output.rstrip("/"))
    output_subdir = (
        args.output_subdir.strip()
        if args.output_subdir is not None
        else BOOST.output_subdir
    )
    sequences = args.sequences or list(BOOST_DEFAULT_SEQUENCES)
    scan_filter = set(args.scan_ids) if args.scan_ids else None

    print(f"Sequences path: {seq_path}")
    print(f"Output: {os.path.join(output_dir, output_subdir)}")
    print(f"Voxel backend: {args.backend}")
    use_icp = not args.no_icp
    use_sor = not args.no_sor
    use_ror = not args.no_ror
    print(
        f"Boost: {'ICP+pose' if use_icp else 'pose-only'}, "
        f"accum_R={args.accumulation_radius} m, output_R={args.output_radius} m, "
        f"window=±{args.window_size}, SOR={use_sor}, ROR={use_ror}"
    )

    generate_sequence_map_boost(
        seq_path=seq_path,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        sequences=sequences,
        backend=args.backend,
        output_subdir=output_subdir,
        scan_ids_filter=scan_filter,
        output_name_suffix=args.name_suffix,
        force=args.force,
        window_half=args.window_size,
        max_gt_points=args.max_gt_points,
        accumulation_radius=args.accumulation_radius,
        output_radius=args.output_radius,
        quiet=args.quiet,
        use_sor=use_sor,
        use_ror=use_ror,
        sor_nb_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        ror_nb_points=args.ror_nb_points,
        ror_radius=args.ror_radius,
        use_icp=use_icp,
        icp_max_iter=args.icp_max_iter,
        icp_threshold=args.icp_threshold,
        icp_downsample=args.icp_downsample,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
