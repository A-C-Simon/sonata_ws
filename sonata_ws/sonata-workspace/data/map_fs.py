#!/usr/bin/env python3
"""
Improved Ground Truth Map Generation (stable + boost ideas)

Features:
- sliding window aggregation
- per-scan cropping (устраняет шум)
- optional ICP (fused-only, стабильный)
- dense voxel near ego (убирает дырки)
- safe ego-based cropping (без пустых облаков)

Output:
ground_truth/{seq}/{scan_id}.npz
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


# =========================
# Calibration + poses
# =========================

def parse_calibration(filename: str) -> dict:
    calib = {}
    if not os.path.exists(filename):
        return calib

    with open(filename) as f:
        for line in f:
            if ":" not in line:
                continue
            key, content = line.split(":", 1)
            values = [float(v) for v in content.strip().split()]
            pose = np.eye(4)
            pose[0, :4] = values[0:4]
            pose[1, :4] = values[4:8]
            pose[2, :4] = values[8:12]
            calib[key.strip()] = pose
    return calib


def load_poses(calib_path, poses_path):
    poses = []

    Tr = None
    if os.path.exists(calib_path):
        calib = parse_calibration(calib_path)
        if "Tr" in calib:
            Tr = calib["Tr"]
            Tr_inv = np.linalg.inv(Tr)

    with open(poses_path) as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0, :4] = values[0:4]
            pose[1, :4] = values[4:8]
            pose[2, :4] = values[8:12]

            if Tr is not None:
                pose = Tr_inv @ pose @ Tr

            poses.append(pose)

    return poses


# =========================
# Voxelization
# =========================

def voxelize_numpy(points, voxel_size):
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    coords = np.floor(points / voxel_size).astype(np.int32)
    unique = np.unique(coords, axis=0)
    return unique.astype(np.float32) * voxel_size + voxel_size / 2


def voxelize_open3d(points, voxel_size):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down.points, dtype=np.float32)


def voxelize(points, voxel_size, backend="open3d"):
    if backend == "open3d":
        return voxelize_open3d(points, voxel_size)
    return voxelize_numpy(points, voxel_size)


# =========================
# ICP (safe version)
# =========================

def apply_icp_fused(all_pts, target_pts):
    import open3d as o3d

    if len(all_pts) < 50 or len(target_pts) < 50:
        return all_pts

    try:
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(all_pts)

        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(target_pts)

        reg = o3d.pipelines.registration.registration_icp(
            src, tgt,
            1.0,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg.transformation
        ones = np.ones((all_pts.shape[0], 1))
        aligned = (T @ np.hstack([all_pts, ones]).T).T[:, :3]

        return aligned

    except Exception:
        return all_pts


# =========================
# Main pipeline
# =========================

def generate_sequence_map(
    seq_path,
    output_dir,
    voxel_size=0.1,
    window=25,
    backend="open3d",
    use_icp=True,
    icp_fused_only=True,
    output_radius=20.0,
    accumulation_radius=15.0,
    dense_core_radius=8.0,
    dense_voxel=0.05,
    max_points=200000
):
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    for seq in sequences:
        print(f"\nProcessing sequence {seq}")

        seq_folder = os.path.join(seq_path, seq)
        velodyne_dir = os.path.join(seq_folder, "velodyne")
        labels_dir = os.path.join(seq_folder, "labels")
        poses_path = os.path.join(seq_folder, "poses.txt")
        calib_path = os.path.join(seq_folder, "calib.txt")

        if not os.path.exists(velodyne_dir):
            continue

        scan_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith(".bin")])
        poses = load_poses(calib_path, poses_path)

        gt_seq_dir = os.path.join(output_dir, "ground_truth", seq)
        os.makedirs(gt_seq_dir, exist_ok=True)

        # -------- load scans --------
        def load_scan_world(idx):
            sf = scan_files[idx]
            pts = np.fromfile(
                os.path.join(velodyne_dir, sf),
                dtype=np.float32
            ).reshape(-1, 4)

            # remove close noise
            dist = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[dist > 3.5]

            # subsample
            if len(pts) > 50000:
                pts = pts[np.random.choice(len(pts), 50000, replace=False)]

            ones = np.ones((pts.shape[0], 1))
            homo = np.hstack([pts[:, :3], ones])
            return (poses[idx] @ homo.T).T[:, :3]

        print("Loading scans...")
        scan_cache = {
            i: load_scan_world(i)
            for i in tqdm(range(len(scan_files)))
        }

        # -------- main loop --------
        for i in tqdm(range(len(scan_files)), desc=f"Sequence {seq}"):

            scan_id = scan_files[i].replace(".bin", "")
            out_path = os.path.join(gt_seq_dir, f"{scan_id}.npz")

            if os.path.exists(out_path):
                continue

            ego = poses[i][:3, 3]

            lo = max(0, i - window)
            hi = min(len(scan_files), i + window + 1)

            local_pts = []

            # 🔥 per-scan crop
            for j in range(lo, hi):
                pts = scan_cache[j]
                center = poses[j][:3, 3]

                dist = np.linalg.norm(pts - center, axis=1)
                pts = pts[dist < accumulation_radius]

                local_pts.append(pts)

            all_pts = np.vstack(local_pts)

            # 🔥 ICP (stable)
            if use_icp and icp_fused_only:
                all_pts = apply_icp_fused(all_pts, scan_cache[i])

            # 🔥 crop around ego (fix empty bug)
            dist = np.linalg.norm(all_pts - ego, axis=1)
            all_pts = all_pts[dist < output_radius]

            # 🔥 dense voxel
            dist = np.linalg.norm(all_pts - ego, axis=1)

            near = all_pts[dist < dense_core_radius]
            far = all_pts[dist >= dense_core_radius]

            voxels = []

            if len(near) > 0:
                voxels.append(voxelize(near, dense_voxel, backend))

            if len(far) > 0:
                voxels.append(voxelize(far, voxel_size, backend))

            if len(voxels) == 0:
                continue

            map_voxel = np.vstack(voxels)

            # transform to scan frame
            pose_inv = np.linalg.inv(poses[i])
            ones = np.ones((map_voxel.shape[0], 1))
            map_scan = (pose_inv @ np.hstack([map_voxel, ones]).T).T[:, :3]

            # subsample
            if len(map_scan) > max_points:
                idx = np.random.choice(len(map_scan), max_points, replace=False)
                map_scan = map_scan[idx]

            np.savez_compressed(out_path, points=map_scan.astype(np.float32))

        print(f"Done sequence {seq}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--backend", default="open3d")

    args = parser.parse_args()

    generate_sequence_map(
        seq_path=args.path,
        output_dir=args.output,
        voxel_size=args.voxel_size,
        window=args.window,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
    