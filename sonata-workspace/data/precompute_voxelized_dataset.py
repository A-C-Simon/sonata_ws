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


def _voxel_centers_merge_duplicates(centers: np.ndarray, voxel_size: float) -> np.ndarray:
    """Merge duplicate voxel centers (e.g. after chunked GPU voxelization)."""
    coords = np.floor(centers / voxel_size).astype(np.int32)
    unique_coords = np.unique(coords, axis=0)
    return (unique_coords.astype(np.float32) * voxel_size + voxel_size / 2).astype(np.float32)


def transform_points_gpu(
    points: np.ndarray,
    T: np.ndarray,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """Transform points (N, 3) by 4x4 matrix T on GPU: p_homo = (x,y,z,1), out = (T @ p_homo)[:3]. Returns (N, 3) float32."""
    try:
        import torch
    except ImportError:
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        pts = np.hstack([points, ones])
        return (T @ pts.T).T[:, :3].astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_t = torch.from_numpy(T.astype(np.float32)).to(dev)
    n = points.shape[0]
    out_list = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end].astype(np.float32)
        ones = np.ones((chunk.shape[0], 1), dtype=np.float32)
        pts = np.hstack([chunk, ones])  # (M, 4)
        pts_t = torch.from_numpy(pts).to(dev, non_blocking=True)
        # (M, 4) @ (4, 4)^T = (M, 4)
        out_t = pts_t @ T_t.T
        out_list.append(out_t[:, :3].cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(out_list).astype(np.float32)


def voxelize_centers_torch(
    points: np.ndarray,
    voxel_size: float,
    device: str = "cuda",
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """Voxelize on GPU (centers only, no labels). Returns (N, 3) float32."""
    try:
        import torch
    except ImportError:
        return (voxelize(points, np.zeros(points.shape[0], dtype=np.int32), voxel_size))[0]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_points = points.shape[0]
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    all_centers = []
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = points[start:end].astype(np.float32)
        pts = torch.from_numpy(chunk).to(dev, non_blocking=True)
        coords = (pts / voxel_size).floor().to(torch.int32)
        del pts
        unique_coords = torch.unique(coords, dim=0)
        del coords
        centers = unique_coords.float() * voxel_size + voxel_size / 2
        del unique_coords
        all_centers.append(centers.cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    merged = np.vstack(all_centers)
    return _voxel_centers_merge_duplicates(merged, voxel_size)


def voxelize_centers_torch_from_tensor(
    pts_tensor,
    voxel_size: float,
    chunk_size: int = 4_000_000,
):
    """Voxelize tensor (N, 3) on same device; returns numpy (K, 3) float32. Keeps work on GPU."""
    import torch
    dev = pts_tensor.device
    n = pts_tensor.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    all_centers = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pts_tensor[start:end]
        coords = (chunk / voxel_size).floor().to(torch.int32)
        unique_coords = torch.unique(coords, dim=0)
        centers = unique_coords.float() * voxel_size + voxel_size / 2
        all_centers.append(centers.cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    merged = np.vstack(all_centers)
    return _voxel_centers_merge_duplicates(merged, voxel_size)


def _voxelize_complete_cpu(complete: np.ndarray, voxel_size: float, backend: str):
    """Voxelize complete cloud (no labels). Returns (centers, zeros labels)."""
    if backend == "torch":
        coord = voxelize_centers_torch(complete, voxel_size)
    else:
        coord, _ = voxelize(complete, np.zeros(complete.shape[0], dtype=np.int32), voxel_size)
    return coord, np.zeros(coord.shape[0], dtype=np.int32)


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
    backend: str = "numpy",
    map_homo_gpu=None,
    device=None,
    gt_npz_path: str = None,
) -> dict:
    """Compute partial + complete voxelized for one frame. Returns dict of numpy arrays.
    When map_homo_gpu and device are provided (backend torch), transform and voxelize stay on GPU.
    When gt_npz_path is provided (per-frame npz from map_from_scans sliding window), complete is
    loaded from that file (already in scan frame), no map_world transform — fast path.
    """
    seq_path = os.path.join(data_path, "sequences", seq)

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

    # Complete: from per-frame npz (fast), or from map_world (GPU or CPU)
    if gt_npz_path and os.path.isfile(gt_npz_path):
        # Fast path: map_from_scans already wrote ground_truth/XX/{scan_id}.npz (scan frame, ≤200k pts)
        gt_data = np.load(gt_npz_path)
        complete = gt_data["points"].astype(np.float32) if "points" in gt_data else gt_data[list(gt_data.keys())[0]].astype(np.float32)
        if complete.ndim == 3:
            complete = complete.reshape(-1, 3)
        complete = complete - scan_center
        complete_coord, complete_labels = _voxelize_complete_cpu(complete, voxel_size, backend)
    elif frame_idx >= poses.shape[0]:
        complete = scan.copy()
        complete_coord, complete_labels = _voxelize_complete_cpu(complete, voxel_size, backend)
    elif map_homo_gpu is not None and device is not None:
        import torch
        Tinv = np.linalg.inv(poses[frame_idx])
        Tinv_gpu = torch.from_numpy(Tinv.astype(np.float32)).to(device)
        scan_center_gpu = torch.from_numpy(scan_center).to(device)
        complete_gpu = (map_homo_gpu @ Tinv_gpu.T)[:, :3] - scan_center_gpu
        complete_coord = voxelize_centers_torch_from_tensor(complete_gpu, voxel_size)
        complete_labels = np.zeros(complete_coord.shape[0], dtype=np.int32)
    else:
        Tinv = np.linalg.inv(poses[frame_idx])
        if backend == "torch" and map_world.shape[0] > 500_000:
            complete = transform_points_gpu(map_world, Tinv) - scan_center
        else:
            ones = np.ones((map_world.shape[0], 1), dtype=np.float32)
            pts = np.hstack([map_world, ones])
            complete = (Tinv @ pts.T).T[:, :3].astype(np.float32) - scan_center
        complete_coord, complete_labels = _voxelize_complete_cpu(complete, voxel_size, backend)

    # Voxelize partial (always CPU for majority-vote labels)
    partial_coord, partial_labels = voxelize(scan, labels_scan, voxel_size)

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each frame id while processing (so you see progress when each frame is slow)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Voxelization backend for 'complete' cloud: numpy (default) or torch (GPU if available)",
    )
    parser.add_argument(
        "--max_map_points",
        type=int,
        default=2_000_000,
        help="Max points from map_world to use per frame (subsample if larger; like map_from_scans). Speeds up and increases GPU use (default 2M).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_path, "voxelized_cache")

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    # CUDA status (so user sees if GPU is used)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok and args.backend == "torch":
            print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA available, using for transform + voxelize)")
        elif args.backend == "torch":
            print("GPU: CUDA not available, --backend torch will use CPU")
        else:
            print("GPU: not used (backend=numpy)")
    except Exception:
        print("GPU: could not check (install PyTorch to use --backend torch)")

    for seq in args.sequences:
        seq_path = os.path.join(args.data_path, "sequences", seq)
        velo_dir = os.path.join(seq_path, "velodyne")
        if not os.path.isdir(velo_dir):
            print(f"Skip seq {seq}: no {velo_dir}")
            continue

        gt_seq_dir = os.path.join(args.data_path, "ground_truth", seq)
        map_npz = os.path.join(gt_seq_dir, "map_world.npz")
        has_map_world = os.path.isfile(map_npz)
        has_per_frame = False
        if os.path.isdir(gt_seq_dir):
            for f in os.listdir(gt_seq_dir):
                if f.endswith(".npz") and f != "map_world.npz":
                    has_per_frame = True
                    break
        if not has_map_world and not has_per_frame:
            print(f"Skip seq {seq}: no {map_npz} and no per-frame npz (run map_from_scans first)")
            continue
        if has_per_frame:
            print(f"  seq {seq}: using per-frame ground_truth/*.npz (fast path, no map_world transform)")

        map_world = None
        map_homo_gpu = None
        device = None
        if has_map_world:
            data = np.load(map_npz)
            map_world = data["points"].astype(np.float32) if "points" in data else data[list(data.keys())[0]].astype(np.float32)
            if map_world.ndim == 3:
                map_world = map_world.reshape(-1, 3)
            n_map = map_world.shape[0]
            if n_map > args.max_map_points:
                rng = np.random.default_rng(int(seq) if seq.isdigit() else 42)
                idx = rng.choice(n_map, args.max_map_points, replace=False)
                map_world = map_world[idx]
                print(f"  seq {seq}: subsampled map {n_map} -> {args.max_map_points} points (--max_map_points)")
            n_map = map_world.shape[0]
            if args.backend == "torch":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        ones = np.ones((n_map, 1), dtype=np.float32)
                        map_homo = np.hstack([map_world, ones])
                        map_homo_gpu = torch.from_numpy(map_homo).to(device, non_blocking=True)
                        print(f"  seq {seq}: map on GPU ({n_map} points)")
                except Exception as e:
                    print(f"  seq {seq}: GPU map failed ({e}), using CPU path")

        pose_path = os.path.join(seq_path, "poses.txt")
        poses = load_poses(pose_path) if os.path.isfile(pose_path) else np.zeros((0, 4, 4))

        scan_files = sorted(f for f in os.listdir(velo_dir) if f.endswith(".bin"))
        out_dir = os.path.join(args.output_dir, seq)
        os.makedirs(out_dir, exist_ok=True)

        for frame_idx, fn in enumerate(tqdm(
            scan_files,
            desc=f"seq {seq}",
            mininterval=1.0,
            miniters=1,
        )):
            scan_id = os.path.splitext(fn)[0]
            out_path = os.path.join(out_dir, f"{scan_id}.npz")
            if args.skip_existing and os.path.isfile(out_path):
                continue
            gt_npz_path = os.path.join(gt_seq_dir, f"{scan_id}.npz")
            if not has_per_frame:
                gt_npz_path = None
            elif not os.path.isfile(gt_npz_path):
                gt_npz_path = None
            if gt_npz_path is None and map_world is None:
                continue
            if args.verbose:
                tqdm.write(f"  [{seq}] frame {frame_idx + 1}/{len(scan_files)} {scan_id}")
            try:
                out = process_frame(
                    args.data_path,
                    seq,
                    scan_id,
                    frame_idx,
                    map_world if map_world is not None else np.zeros((0, 3), dtype=np.float32),
                    poses,
                    args.voxel_size,
                    args.max_points,
                    backend=args.backend,
                    map_homo_gpu=map_homo_gpu,
                    device=device,
                    gt_npz_path=gt_npz_path,
                )
                np.savez_compressed(out_path, **out)
                total += 1
            except Exception as e:
                tqdm.write(f"Skip {seq}/{scan_id}: {e}")

        if map_homo_gpu is not None:
            del map_homo_gpu
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"Done. Saved {total} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
