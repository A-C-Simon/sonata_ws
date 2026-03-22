#!/usr/bin/env python3
"""
Refined ground-truth maps: same merge + voxel as map_from_scans.py, then Open3D cleanup.

What is implemented (simple, safe to compare with baseline):
  - remove_statistical_outlier  (optional)
  - remove_radius_outlier       (optional)
  - optional extra radius pass (--remove_floaters) for isolated points in void
  - optional DBSCAN noise removal (--dbscan_noise, Open3D)

Optional:
  - --icp  incremental ICP in world frame (Open3D) before voxelization
  - --icp_fast  preset (smaller history/scales, crop, point caps, early RMSE stop)
  - ray-cast visibility / temporal weighting

By default, results are written under this repo (not next to SemanticKITTI):
  <sonata-workspace>/gt_maps_refined/ground_truth/{seq}/{scan_id}.npz

Override with -o / --output_subdir if needed.

Example (seq 00, scan 000000):
  cd /path/to/sonata-workspace
  python data/new_map_from_scans.py \\
    -p ~/dataset/SemanticKITTI/dataset/sequences \\
    --scan_ids 000000 --sequences 00 --force

Speed (typical):
  -b open3d or -b auto  (faster voxelize than pure numpy on large merges)
  --icp --icp_fast      (cheaper ICP; see merge_window_points_icp)
  --quiet               (less tqdm overhead on long runs)
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

# Import voxelize / load_poses from sibling module
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DATA_DIR)
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)
from map_from_scans import load_poses, voxelize  # noqa: E402

# Default export root inside sonata-workspace (keeps GT separate from dataset tree)
_DEFAULT_GT_EXPORT = os.path.join(_REPO_ROOT, "gt_maps_refined")


def refine_points_scipy(
    points: np.ndarray,
    use_statistical: bool,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    use_radius: bool,
    radius_nb_points: int,
    radius: float,
) -> np.ndarray:
    """
    Same idea as Open3D statistical + radius outlier removal, using scipy (no libX11).
    """
    from scipy.spatial import cKDTree

    if points.shape[0] == 0:
        return points.astype(np.float32)

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    if use_statistical and n > 2:
        k = min(stat_nb_neighbors, n - 1)
        if k >= 1:
            tree = cKDTree(pts)
            # k+1 neighbors: first is self (dist 0)
            try:
                dists, _ = tree.query(pts, k=k + 1, workers=-1)
            except TypeError:
                dists, _ = tree.query(pts, k=k + 1)
            mean_d = dists[:, 1:].mean(axis=1)
            mu, sigma = float(mean_d.mean()), float(mean_d.std()) + 1e-9
            keep = mean_d <= mu + stat_std_ratio * sigma
            pts = pts[keep]
            n = pts.shape[0]

    if use_radius and n > 0:
        tree = cKDTree(pts)
        try:
            neighbors = tree.query_ball_point(pts, r=radius, workers=-1)
        except TypeError:
            neighbors = tree.query_ball_point(pts, r=radius)
        counts = np.array([len(neighbors[i]) for i in range(len(neighbors))])
        keep = counts >= radius_nb_points
        pts = pts[keep]

    return pts.astype(np.float32)


def refine_points_scipy_floater(
    points: np.ndarray,
    floater_radius: float,
    floater_min_neighbors: int,
) -> np.ndarray:
    """Second-pass radius outlier (sparse / floating points) without Open3D."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    tree = cKDTree(pts)
    try:
        neighbors = tree.query_ball_point(pts, r=floater_radius, workers=-1)
    except TypeError:
        neighbors = tree.query_ball_point(pts, r=floater_radius)
    counts = np.array([len(neighbors[i]) for i in range(len(neighbors))])
    keep = counts >= floater_min_neighbors
    pts = pts[keep]
    return pts.astype(np.float32)


def refine_points(
    points: np.ndarray,
    use_statistical: bool,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    use_radius: bool,
    radius_nb_points: int,
    radius: float,
    use_floater_pass: bool = False,
    floater_radius: float = 0.5,
    floater_min_neighbors: int = 10,
    use_dbscan_noise: bool = False,
    dbscan_eps: float = 0.45,
    dbscan_min_points: int = 10,
) -> np.ndarray:
    """
    Statistical + radius outlier filters; Open3D if available, else scipy.

    Optional (after SOR + ROR):
      - Floater pass: stricter radius outlier (few neighbors within a ball → drop).
        Cuts isolated points in empty space between surfaces.
      - DBSCAN (Open3D only): remove cluster label -1 (noise).
    """
    if points.shape[0] == 0:
        return points

    try:
        import open3d as o3d
    except (ImportError, OSError):
        out = refine_points_scipy(
            points,
            use_statistical,
            stat_nb_neighbors,
            stat_std_ratio,
            use_radius,
            radius_nb_points,
            radius,
        )
        if use_floater_pass and len(out) > 0:
            out = refine_points_scipy_floater(
                out, floater_radius, floater_min_neighbors
            )
        if use_dbscan_noise and len(out) > 0:
            pass
        return out

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if use_statistical:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=stat_nb_neighbors,
            std_ratio=stat_std_ratio,
        )
    if use_radius and len(pcd.points) > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=radius_nb_points,
            radius=radius,
        )

    if use_floater_pass and len(pcd.points) > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=floater_min_neighbors,
            radius=floater_radius,
        )

    if use_dbscan_noise and len(pcd.points) > 0:
        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                print_progress=False,
            )
        )
        if labels.size > 0:
            idx_keep = np.flatnonzero(labels >= 0)
            if idx_keep.size > 0:
                pcd = pcd.select_by_index(idx_keep.tolist())
            else:
                pcd.points = o3d.utility.Vector3dVector(
                    np.zeros((0, 3), dtype=np.float64)
                )

    if len(pcd.points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pcd.points, dtype=np.float32)


def _icp_estimation(o3d, point_to_plane: bool, robust: bool):
    """Point-to-plane (optional robust Tukey loss) or point-to-point fallback."""
    if not point_to_plane:
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()
    if robust:
        try:
            loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(
                loss
            )
        except (AttributeError, TypeError):
            pass
    return o3d.pipelines.registration.TransformationEstimationPointToPlane()


def _crop_points_radius(pts: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if radius is None or radius <= 0 or pts.shape[0] == 0:
        return pts
    d = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
    return pts[d <= radius]


def _icp_subsample(pts: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Random subsample for faster ICP; full cloud is still used for applying T."""
    if max_points is None or max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    idx = rng.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]


def merge_window_points_icp(
    scans_world: list,
    icp_voxel: float = 0.25,
    icp_threshold: float = 1.0,
    icp_max_iter: int = 30,
    icp_target_history: int = 8,
    icp_scales: list = None,
    icp_point_to_plane: bool = True,
    icp_robust: bool = False,
    icp_crop_radius: float = None,
    icp_legacy: bool = False,
    icp_src_max_points: int = None,
    icp_tgt_max_points: int = None,
    icp_normals_finest_only: bool = True,
    icp_early_stop_rmse: float = None,
    icp_rng_seed: int = 42,
) -> np.ndarray:
    """
    Merge scans in world coordinates with incremental ICP.

    Default (icp_legacy=False):
      - Target = last ``icp_target_history`` *aligned* scans (not full fused) to limit drift.
      - Multi-scale voxel sizes ``icp_scales`` (default ~0.5 → 0.2 → 0.08 m).
      - Point-to-plane ICP with estimated normals (better for roads / planes).
      - Optional Tukey robust loss; optional spherical crop around source centroid.

    Speed (optional):
      - ``icp_src_max_points`` / ``icp_tgt_max_points``: random cap on points *for ICP only*
        (full-resolution ``src_raw`` still gets the final transform).
      - ``icp_normals_finest_only``: coarse scales use point-to-point; normals only on last scale.
      - ``icp_early_stop_rmse``: stop multi-scale loop when RMSE is already small.

    icp_legacy=True: previous single-scale point-to-point vs full fused (old behavior).
    """
    import open3d as o3d

    rng = np.random.default_rng(icp_rng_seed)

    if len(scans_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(scans_world) == 1:
        return np.asarray(scans_world[0], dtype=np.float32)

    if icp_legacy:
        fused = np.asarray(scans_world[0], dtype=np.float64)
        for k in range(1, len(scans_world)):
            src = np.asarray(scans_world[k], dtype=np.float64)
            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(src)
            pcd_tgt = o3d.geometry.PointCloud()
            pcd_tgt.points = o3d.utility.Vector3dVector(fused)
            src_d = pcd_src.voxel_down_sample(icp_voxel)
            tgt_d = pcd_tgt.voxel_down_sample(icp_voxel)
            if len(src_d.points) < 20 or len(tgt_d.points) < 20:
                fused = np.vstack([fused, src])
                continue
            reg = o3d.pipelines.registration.registration_icp(
                src_d,
                tgt_d,
                icp_threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=icp_max_iter
                ),
            )
            T = np.asarray(reg.transformation, dtype=np.float64)
            n_h = src.shape[0]
            homo = np.hstack([src, np.ones((n_h, 1))])
            aligned = (T @ homo.T).T[:, :3]
            fused = np.vstack([fused, aligned])
        return fused.astype(np.float32)

    scales = icp_scales if icp_scales is not None else [0.5, 0.2, 0.08]
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter)

    aligned_list = [np.asarray(scans_world[0], dtype=np.float64)]
    fused = np.asarray(scans_world[0], dtype=np.float64)

    for k in range(1, len(scans_world)):
        src_raw = np.asarray(scans_world[k], dtype=np.float64)
        if icp_target_history <= 0:
            start = 0
        else:
            start = max(0, k - icp_target_history)
        tgt_pts = np.vstack(aligned_list[start:k])
        if tgt_pts.shape[0] < 20 or src_raw.shape[0] < 20:
            aligned_list.append(src_raw.copy())
            fused = np.vstack([fused, src_raw])
            continue

        center = np.mean(src_raw, axis=0)
        src_work = src_raw
        tgt_work = tgt_pts
        if icp_crop_radius is not None and icp_crop_radius > 0:
            src_work = _crop_points_radius(src_work, center, icp_crop_radius)
            tgt_work = _crop_points_radius(tgt_work, center, icp_crop_radius)
            if src_work.shape[0] < 20 or tgt_work.shape[0] < 20:
                src_work = src_raw
                tgt_work = tgt_pts

        src_work = _icp_subsample(src_work, icp_src_max_points, rng)
        tgt_work = _icp_subsample(tgt_work, icp_tgt_max_points, rng)
        if src_work.shape[0] < 20 or tgt_work.shape[0] < 20:
            aligned_list.append(src_raw.copy())
            fused = np.vstack([fused, src_raw])
            continue

        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(src_work)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(tgt_work)

        T = np.eye(4)
        ref_scale = scales[0]
        n_scales = len(scales)
        for si, v in enumerate(scales):
            src_ds = pcd_src.voxel_down_sample(v)
            tgt_ds = pcd_tgt.voxel_down_sample(v)
            if len(src_ds.points) < 15 or len(tgt_ds.points) < 15:
                break
            th = icp_threshold * (v / ref_scale)
            th = max(th, v * 3.0)

            use_plane = icp_point_to_plane and (
                not icp_normals_finest_only or si == n_scales - 1
            )
            if use_plane:
                rad_n = max(v * 3.0, 0.3)
                src_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=rad_n, max_nn=40
                    )
                )
                tgt_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=rad_n, max_nn=40
                    )
                )
                est = _icp_estimation(o3d, True, icp_robust)
            else:
                est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

            reg = o3d.pipelines.registration.registration_icp(
                src_ds,
                tgt_ds,
                th,
                T,
                est,
                crit,
            )
            T = np.asarray(reg.transformation, dtype=np.float64)

            if icp_early_stop_rmse is not None:
                rmse = getattr(reg, "inlier_rmse", None)
                if rmse is not None and rmse < icp_early_stop_rmse:
                    break

        n_h = src_raw.shape[0]
        homo = np.hstack([src_raw, np.ones((n_h, 1))])
        aligned = (T @ homo.T).T[:, :3]
        aligned_list.append(aligned.copy())
        fused = np.vstack([fused, aligned])

    return fused.astype(np.float32)


def generate_sequence_map_refined(
    seq_path: str,
    output_dir: str,
    voxel_size: float = 0.1,
    sequences: list = None,
    backend: str = "numpy",
    output_subdir: str = "ground_truth",
    scan_ids_filter: set = None,
    output_name_suffix: str = "",
    force: bool = False,
    use_statistical: bool = True,
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 1.0,
    use_radius: bool = True,
    radius_nb_points: int = 5,
    radius: float = 0.3,
    window: int = 25,
    max_gt_points: int = 200000,
    use_icp: bool = False,
    icp_voxel: float = 0.25,
    icp_threshold: float = 1.0,
    icp_max_iter: int = 30,
    icp_target_history: int = 8,
    icp_scales: list = None,
    icp_point_to_plane: bool = True,
    icp_robust: bool = False,
    icp_crop_radius: float = None,
    icp_legacy: bool = False,
    icp_src_max_points: int = None,
    icp_tgt_max_points: int = None,
    icp_normals_finest_only: bool = True,
    icp_early_stop_rmse: float = None,
    icp_rng_seed: int = 42,
    use_floater_pass: bool = False,
    floater_radius: float = 0.5,
    floater_min_neighbors: int = 10,
    use_dbscan_noise: bool = False,
    dbscan_eps: float = 0.45,
    dbscan_min_points: int = 10,
    quiet: bool = False,
) -> None:
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

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

        def load_scan_world(idx):
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
            pts = pts[dist > 3.5]
            if len(pts) > 50000:
                idx_sub = np.random.choice(len(pts), 50000, replace=False)
                pts = pts[idx_sub]
            ones = np.ones((pts.shape[0], 1))
            homo = np.hstack([pts[:, :3], ones])
            return (poses[idx] @ homo.T).T[:, :3]

        if scan_ids_filter is not None:
            needed = set()
            for i in indices:
                lo = max(0, i - window)
                hi = min(n_scans, i + window + 1)
                needed.update(range(lo, hi))
            to_load = sorted(needed)
            print(
                f"  Loading {len(to_load)} scans "
                f"(window {window}; not all {n_scans}) for refined GT..."
            )
        else:
            to_load = list(range(n_scans))
            print(f"  Loading {n_scans} scans...")

        scan_cache = {}
        for idx in tqdm(
            to_load, desc=f"Loading {seq}", leave=False, disable=quiet
        ):
            scan_cache[idx] = load_scan_world(idx)

        for i in tqdm(indices, desc=f"Sequence {seq} (refined)", disable=quiet):
            scan_id = scan_files[i].replace(".bin", "")
            base_name = f"{scan_id}{output_name_suffix}" if output_name_suffix else scan_id
            out_path = os.path.join(gt_seq_dir, f"{base_name}.npz")
            if os.path.exists(out_path) and not force:
                continue

            lo = max(0, i - window)
            hi = min(len(scan_files), i + window + 1)
            local_pts = [scan_cache[j] for j in range(lo, hi)]
            if use_icp:
                try:
                    all_pts = merge_window_points_icp(
                        local_pts,
                        icp_voxel=icp_voxel,
                        icp_threshold=icp_threshold,
                        icp_max_iter=icp_max_iter,
                        icp_target_history=icp_target_history,
                        icp_scales=icp_scales,
                        icp_point_to_plane=icp_point_to_plane,
                        icp_robust=icp_robust,
                        icp_crop_radius=icp_crop_radius,
                        icp_legacy=icp_legacy,
                        icp_src_max_points=icp_src_max_points,
                        icp_tgt_max_points=icp_tgt_max_points,
                        icp_normals_finest_only=icp_normals_finest_only,
                        icp_early_stop_rmse=icp_early_stop_rmse,
                        icp_rng_seed=icp_rng_seed,
                    )
                except (ImportError, OSError, RuntimeError) as e:
                    print(f"  ICP failed ({e}), falling back to pose-only merge.")
                    all_pts = np.vstack(local_pts)
            else:
                all_pts = np.vstack(local_pts)
            map_voxel = voxelize(all_pts, voxel_size, backend=backend)
            del all_pts
            pose_inv = np.linalg.inv(poses[i])
            ones = np.ones((map_voxel.shape[0], 1))
            map_scan = (pose_inv @ np.hstack([map_voxel, ones]).T).T[:, :3]
            del map_voxel

            map_scan = refine_points(
                map_scan,
                use_statistical=use_statistical,
                stat_nb_neighbors=stat_nb_neighbors,
                stat_std_ratio=stat_std_ratio,
                use_radius=use_radius,
                radius_nb_points=radius_nb_points,
                radius=radius,
                use_floater_pass=use_floater_pass,
                floater_radius=floater_radius,
                floater_min_neighbors=floater_min_neighbors,
                use_dbscan_noise=use_dbscan_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
            )

            if len(map_scan) > max_gt_points:
                idx_sub = np.random.choice(len(map_scan), max_gt_points, replace=False)
                map_scan = map_scan[idx_sub]

            np.savez_compressed(out_path, points=map_scan.astype(np.float32))

        del scan_cache
        rel = os.path.join(output_subdir, seq) if output_subdir else seq
        print(f"Saved refined ground truth for sequence {seq} -> {rel}/")


def main():
    _default_dataset = os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset")
    _default_sequences = os.path.join(_default_dataset, "sequences")
    parser = argparse.ArgumentParser(
        description="GT maps with optional Open3D outlier removal (compare to map_from_scans)"
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=_default_sequences,
        help="SemanticKITTI sequences folder (velodyne, poses, ...)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=_DEFAULT_GT_EXPORT,
        help=f"Where to write npz (default: {_DEFAULT_GT_EXPORT} under sonata-workspace)",
    )
    parser.add_argument("--voxel_size", "-v", type=float, default=0.1)
    parser.add_argument(
        "--sequences", "-s", type=str, nargs="+", default=None,
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["numpy", "open3d", "torch", "auto"],
        default="numpy",
        help="Voxelize: numpy (default), open3d (often faster), torch (GPU), auto=open3d if ok",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No tqdm bars (poses / loading / sequence loop)",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="ground_truth",
        help="Subfolder under -o (default: ground_truth). Use \"\" to write <output>/<seq>/ directly.",
    )
    parser.add_argument(
        "--scan_ids",
        nargs="+",
        default=None,
        help="Only these scan ids (e.g. 000000) for quick tests",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Append to output basename, e.g. _icp -> 000000_icp.npz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing npz",
    )
    parser.add_argument("--no_statistical", action="store_true")
    parser.add_argument("--no_radius", action="store_true")
    parser.add_argument("--stat_nb", type=int, default=20)
    parser.add_argument("--stat_std", type=float, default=1.0)
    parser.add_argument("--radius_nb", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument(
        "--remove_floaters",
        action="store_true",
        help=(
            "After SOR/ROR, extra radius outlier pass to drop sparse points "
            "in empty space (see --floater_radius, --floater_min_neighbors)"
        ),
    )
    parser.add_argument(
        "--floater_radius",
        type=float,
        default=0.5,
        help="Radius (m) for floater pass; need >= floater_min_neighbors neighbors inside",
    )
    parser.add_argument(
        "--floater_min_neighbors",
        type=int,
        default=10,
        help="Min neighbor count within floater_radius to keep a point",
    )
    parser.add_argument(
        "--dbscan_noise",
        action="store_true",
        help=(
            "After other filters, remove Open3D DBSCAN noise (label -1); "
            "ignored without Open3D"
        ),
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.45,
        help="DBSCAN neighborhood radius (m) for noise removal",
    )
    parser.add_argument(
        "--dbscan_min_points",
        type=int,
        default=10,
        help="DBSCAN min_points (core point)",
    )
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument(
        "--icp",
        action="store_true",
        help="Incremental ICP merge in world frame before voxel (needs Open3D)",
    )
    parser.add_argument(
        "--icp_voxel",
        type=float,
        default=0.25,
        help="Voxel size for ICP downsample (meters)",
    )
    parser.add_argument(
        "--icp_threshold",
        type=float,
        default=1.0,
        help="ICP max correspondence distance (meters)",
    )
    parser.add_argument(
        "--icp_max_iter",
        type=int,
        default=30,
        help="ICP max iterations per pairwise alignment",
    )
    parser.add_argument(
        "--icp_target_history",
        type=int,
        default=8,
        help="ICP target = last K aligned scans in window (0 = all previous, like old fused)",
    )
    parser.add_argument(
        "--icp_scales",
        nargs="+",
        type=float,
        default=None,
        help="Multi-scale voxel sizes (m), e.g. 0.5 0.2 0.08 (default: 0.5 0.2 0.08)",
    )
    parser.add_argument(
        "--icp_no_point_to_plane",
        action="store_true",
        help="Use point-to-point ICP instead of point-to-plane",
    )
    parser.add_argument(
        "--icp_robust",
        action="store_true",
        help="Tukey robust loss for point-to-plane (if supported by Open3D)",
    )
    parser.add_argument(
        "--icp_crop_radius",
        type=float,
        default=None,
        help="Optional crop radius (m) around scan centroid for ICP only",
    )
    parser.add_argument(
        "--icp_legacy",
        action="store_true",
        help="Legacy: single-scale point-to-point vs full fused map",
    )
    parser.add_argument(
        "--icp_src_max_points",
        type=int,
        default=None,
        help="Random cap on source points for ICP only (full scan still transformed)",
    )
    parser.add_argument(
        "--icp_tgt_max_points",
        type=int,
        default=None,
        help="Random cap on target points for ICP only",
    )
    parser.add_argument(
        "--icp_normals_every_scale",
        action="store_true",
        help="Estimate normals at every scale (slower; default: normals only on finest scale)",
    )
    parser.add_argument(
        "--icp_early_stop_rmse",
        type=float,
        default=None,
        help="Stop multi-scale ICP early when inlier RMSE is below this (meters)",
    )
    parser.add_argument(
        "--icp_rng_seed",
        type=int,
        default=42,
        help="RNG seed for ICP subsampling (reproducibility)",
    )
    parser.add_argument(
        "--icp_fast",
        action="store_true",
        help=(
            "Preset: history=5, iter=12, scales 0.3 0.1, crop=30m, "
            "src<=20k tgt<=50k, early_stop_rmse=0.02 (overrides only unset options)"
        ),
    )

    args = parser.parse_args()

    if args.icp_fast:
        args.icp_target_history = 5
        args.icp_max_iter = 12
        if args.icp_scales is None:
            args.icp_scales = [0.3, 0.1]
        if args.icp_crop_radius is None:
            args.icp_crop_radius = 30.0
        if args.icp_src_max_points is None:
            args.icp_src_max_points = 20000
        if args.icp_tgt_max_points is None:
            args.icp_tgt_max_points = 50000
        if args.icp_early_stop_rmse is None:
            args.icp_early_stop_rmse = 0.02

    seq_path = args.path.rstrip("/")
    output_dir = os.path.abspath(args.output.rstrip("/"))
    output_subdir = args.output_subdir.strip() if args.output_subdir is not None else "ground_truth"
    sequences = args.sequences
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    scan_filter = set(args.scan_ids) if args.scan_ids else None

    print(f"Sequences path: {seq_path}")
    print(
        "Output dir:",
        os.path.join(output_dir, output_subdir) if output_subdir else output_dir,
    )
    print(f"Voxel backend: {args.backend}")
    print(f"Statistical outlier: {not args.no_statistical}, radius: {not args.no_radius}")
    print(
        f"Floater pass: {args.remove_floaters}, DBSCAN noise trim: {args.dbscan_noise}"
    )
    print(f"ICP merge: {args.icp}")
    if args.icp:
        print(
            f"  ICP: history={args.icp_target_history}, scales={args.icp_scales}, "
            f"p2plane={not args.icp_no_point_to_plane}, robust={args.icp_robust}, "
            f"crop={args.icp_crop_radius}, legacy={args.icp_legacy}, "
            f"src_max={args.icp_src_max_points}, tgt_max={args.icp_tgt_max_points}, "
            f"normals_finest_only={not args.icp_normals_every_scale}, "
            f"early_stop_rmse={args.icp_early_stop_rmse}"
        )

    generate_sequence_map_refined(
        seq_path=seq_path,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        sequences=sequences,
        backend=args.backend,
        output_subdir=output_subdir,
        scan_ids_filter=scan_filter,
        output_name_suffix=args.name_suffix,
        force=args.force,
        use_statistical=not args.no_statistical,
        stat_nb_neighbors=args.stat_nb,
        stat_std_ratio=args.stat_std,
        use_radius=not args.no_radius,
        radius_nb_points=args.radius_nb,
        radius=args.radius,
        window=args.window,
        use_icp=args.icp,
        icp_voxel=args.icp_voxel,
        icp_threshold=args.icp_threshold,
        icp_max_iter=args.icp_max_iter,
        icp_target_history=args.icp_target_history,
        icp_scales=args.icp_scales,
        icp_point_to_plane=not args.icp_no_point_to_plane,
        icp_robust=args.icp_robust,
        icp_crop_radius=args.icp_crop_radius,
        icp_legacy=args.icp_legacy,
        icp_src_max_points=args.icp_src_max_points,
        icp_tgt_max_points=args.icp_tgt_max_points,
        icp_normals_finest_only=not args.icp_normals_every_scale,
        icp_early_stop_rmse=args.icp_early_stop_rmse,
        icp_rng_seed=args.icp_rng_seed,
        use_floater_pass=args.remove_floaters,
        floater_radius=args.floater_radius,
        floater_min_neighbors=args.floater_min_neighbors,
        use_dbscan_noise=args.dbscan_noise,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min_points,
        quiet=args.quiet,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
