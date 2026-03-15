"""
Precompute voxelized (partial + complete) samples once, after map_from_scans.

Does not require data.semantickitti: uses SemanticKITTI layout on disk only.
Saves one small .npz per (sequence, scan_id). Training with --voxelized_cache_dir
then loads these files (no map_world transform or voxelization at train time).

Run from repo root after map_from_scans has produced ground_truth/XX/map_world.npz:

  python data/precompute_voxelized_dataset.py \
    --data_path /path/to/SemanticKITTI/dataset \
    --output_dir /path/to/SemanticKITTI/dataset/voxelized_cache \
    --voxel_size 0.05 \
    --max_points 20000 \
    --sequences 00 01 02 03 04 05 06 07 09 10
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same as SemanticKITTI: map raw SemanticKITTI labels to 20 learning classes
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14,
    52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
    252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}


def load_poses(pose_path: str) -> np.ndarray:
    """Load KITTI poses: one 3x4 matrix per line (12 floats). Return (N, 4, 4)."""
    poses = []
    with open(pose_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) < 12:
                continue
            T = np.eye(4)
            T[0, :4] = vals[0:4]
            T[1, :4] = vals[4:8]
            T[2, :4] = vals[8:12]
            poses.append(T)
    return np.array(poses) if poses else np.zeros((0, 4, 4))


def load_scan(bin_path: str) -> np.ndarray:
    """Load velodyne .bin: float32, (N, 4) -> xyz (N, 3)."""
    x = np.fromfile(bin_path, dtype=np.float32)
    x = x.reshape(-1, 4)
    return x[:, :3].astype(np.float32)


def load_labels(label_path: str, n_points: int, apply_learning_map: bool = True) -> np.ndarray:
    """Load .label (uint32), mask to 16-bit, optionally map to 20 classes, return int32 (n_points,)."""
    if not os.path.exists(label_path):
        return np.zeros(n_points, dtype=np.int32)
    x = np.fromfile(label_path, dtype=np.uint32)
    x = x.reshape(-1)
    x = (x & 0xFFFF).astype(np.int32)
    if len(x) != n_points:
        return np.zeros(n_points, dtype=np.int32)
    if apply_learning_map:
        x = np.vectorize(lambda v: LEARNING_MAP.get(int(v), 0))(x)
    return x


def voxelize(points: np.ndarray, labels: np.ndarray, voxel_size: float):
    """Unique voxels -> centers (float32), majority-vote labels (int32)."""
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    centers = (unique_voxels * voxel_size + voxel_size / 2).astype(np.float32)
    out_labels = np.zeros(len(unique_voxels), dtype=np.int32)
    if labels.size and np.any(labels):
        n_cl = int(labels.max()) + 1
        counts = np.zeros((len(unique_voxels), n_cl), dtype=np.int32)
        np.add.at(counts, (inverse, labels), 1)
        out_labels = counts.argmax(axis=1).astype(np.int32)
    return centers, out_labels


def height_to_color(z: np.ndarray) -> np.ndarray:
    """ (N,) -> (N, 3) RGB from height."""
    z = np.asarray(z, dtype=np.float32)
    zmin, zmax = z.min(), z.max()
    if zmax - zmin < 1e-6:
        znorm = np.zeros_like(z)
    else:
        znorm = (z - zmin) / (zmax - zmin)
    out = np.zeros((z.shape[0], 3), dtype=np.float32)
    out[:, 0] = znorm
    out[:, 1] = 1 - np.abs(znorm - 0.5) * 2
    out[:, 2] = 1 - znorm
    return out


def process_frame(
    data_path: str,
    seq: str,
    scan_id: str,
    frame_idx: int,
    map_world: np.ndarray,
    poses: np.ndarray,
    voxel_size: float,
    max_points: int,
) -> dict:
    """Compute partial + complete voxelized for one frame. Returns dict of numpy arrays."""
    seq_path = os.path.join(data_path, "sequences", seq)
    gt_path = os.path.join(data_path, "ground_truth", seq)

    # Scan
    bin_path = os.path.join(seq_path, "velodyne", f"{scan_id}.bin")
    scan = load_scan(bin_path)
    n_scan = scan.shape[0]
    labels_scan = load_labels(
        os.path.join(seq_path, "labels", f"{scan_id}.label"), n_scan, apply_learning_map=True
    )

    # Center
    scan_center = scan.mean(axis=0).astype(np.float32)
    scan = scan - scan_center

    # Map in scan frame
    if frame_idx >= poses.shape[0]:
        complete = scan.copy()
    else:
        Tinv = np.linalg.inv(poses[frame_idx])
        ones = np.ones((map_world.shape[0], 1), dtype=np.float32)
        pts = np.hstack([map_world, ones])
        complete = (Tinv @ pts.T).T[:, :3].astype(np.float32) - scan_center

    # Voxelize
    partial_coord, partial_labels = voxelize(scan, labels_scan, voxel_size)
    complete_coord, complete_labels = voxelize(complete, np.zeros(complete.shape[0], dtype=np.int32), voxel_size)

    # Subsample
    if partial_coord.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(partial_coord.shape[0], max_points, replace=False)
        partial_coord = partial_coord[idx]
        partial_labels = partial_labels[idx]
    if complete_coord.shape[0] > max_points:
        idx = np.random.default_rng(43).choice(complete_coord.shape[0], max_points, replace=False)
        complete_coord = complete_coord[idx]
        complete_labels = complete_labels[idx]

    # Colors and normals
    partial_color = height_to_color(partial_coord[:, 2])
    complete_color = height_to_color(complete_coord[:, 2])
    partial_normal = np.zeros_like(partial_coord)
    complete_normal = np.zeros_like(complete_coord)

    return {
        "partial_coord": partial_coord,
        "partial_color": partial_color,
        "partial_normal": partial_normal,
        "partial_labels": partial_labels,
        "complete_coord": complete_coord,
        "complete_color": complete_color,
        "complete_normal": complete_normal,
        "complete_labels": complete_labels,
        "scan_center": scan_center,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Precompute voxelized cache (one npz per frame) after map_from_scans"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="SemanticKITTI dataset root (contains sequences/, ground_truth/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for npz cache (default: <data_path>/voxelized_cache)",
    )
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        help="Sequence IDs to precompute",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip frames that already have an npz",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_path, "voxelized_cache")

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for seq in args.sequences:
        seq_path = os.path.join(args.data_path, "sequences", seq)
        velo_dir = os.path.join(seq_path, "velodyne")
        if not os.path.isdir(velo_dir):
            print(f"Skip seq {seq}: no {velo_dir}")
            continue

        map_npz = os.path.join(args.data_path, "ground_truth", seq, "map_world.npz")
        if not os.path.isfile(map_npz):
            print(f"Skip seq {seq}: no {map_npz} (run map_from_scans first)")
            continue

        data = np.load(map_npz)
        map_world = data["points"].astype(np.float32) if "points" in data else data[list(data.keys())[0]].astype(np.float32)
        if map_world.ndim == 3:
            map_world = map_world.reshape(-1, 3)

        pose_path = os.path.join(seq_path, "poses.txt")
        poses = load_poses(pose_path) if os.path.isfile(pose_path) else np.zeros((0, 4, 4))

        scan_files = sorted(f for f in os.listdir(velo_dir) if f.endswith(".bin"))
        out_dir = os.path.join(args.output_dir, seq)
        os.makedirs(out_dir, exist_ok=True)

        for frame_idx, fn in enumerate(tqdm(scan_files, desc=f"seq {seq}")):
            scan_id = os.path.splitext(fn)[0]
            out_path = os.path.join(out_dir, f"{scan_id}.npz")
            if args.skip_existing and os.path.isfile(out_path):
                continue
            try:
                out = process_frame(
                    args.data_path,
                    seq,
                    scan_id,
                    frame_idx,
                    map_world,
                    poses,
                    args.voxel_size,
                    args.max_points,
                )
                np.savez_compressed(out_path, **out)
                total += 1
            except Exception as e:
                tqdm.write(f"Skip {seq}/{scan_id}: {e}")

    print(f"Done. Saved {total} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
