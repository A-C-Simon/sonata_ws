#!/usr/bin/env python3
"""
Plan B: build voxelized cache for SemanticKITTI (faster training).

  1) map_from_scans with --save_only_map_world → ground_truth/XX/map_world.npz
  2) precompute_voxelized_dataset → voxelized_cache/XX/{scan_id}.npz

Then train with --voxelized_cache_dir pointing to the cache.

Run from project root:
  python scripts/run_voxelized_cache_semantickitti.py --data_path /path/to/SemanticKITTI/dataset
  # If map_world already exists:
  python scripts/run_voxelized_cache_semantickitti.py --data_path /path/to/dataset --skip_map
"""

import argparse
import os
import subprocess
from typing import List


def run(cmd: List[str], cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build voxelized cache for SemanticKITTI (plan B)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"),
        help="SemanticKITTI dataset root (contains sequences/, will create ground_truth/, voxelized_cache/)",
    )
    parser.add_argument(
        "--skip_map",
        action="store_true",
        help="Skip map_from_scans; only run precompute (use when map_world.npz already exists)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=[f"{i:02d}" for i in range(11)],
        help="Sequence IDs (default: 00 01 ... 10)",
    )
    parser.add_argument(
        "--voxel_size_map",
        type=float,
        default=0.1,
        help="Voxel size for map_from_scans (default 0.1)",
    )
    parser.add_argument(
        "--voxel_size_cache",
        type=float,
        default=0.05,
        help="Voxel size for precomputed cache, must match training (default 0.05)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["numpy", "open3d", "torch"],
        help="Voxelization backend for map_from_scans",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(args.data_path)
    seq_path = os.path.join(data_path, "sequences")
    voxelized_cache = os.path.join(data_path, "voxelized_cache")

    if not args.skip_map:
        print("=== 1) map_from_scans (save_only_map_world) ===")
        run(
            [
                "python",
                "data/map_from_scans.py",
                "--path",
                seq_path,
                "--output",
                data_path,
                "--voxel_size",
                str(args.voxel_size_map),
                "--backend",
                args.backend,
                "--save_only_map_world",
                "--sequences",
                *args.sequences,
            ],
            cwd=repo_root,
        )
    else:
        print("=== 1) Skipping map_from_scans (--skip_map) ===")

    print("\n=== 2) precompute_voxelized_dataset ===")
    run(
        [
            "python",
            "data/precompute_voxelized_dataset.py",
            "--data_path",
            data_path,
            "--output_dir",
            voxelized_cache,
            "--voxel_size",
            str(args.voxel_size_cache),
            "--sequences",
            *args.sequences,
        ],
        cwd=repo_root,
    )

    print("\n" + "=" * 60)
    print("Plan B cache ready. Train with:")
    print("=" * 60)
    print(
        f"  python training/train_diffusion.py \\\n"
        f"    --data_path {data_path} \\\n"
        f"    --voxelized_cache_dir {voxelized_cache} \\\n"
        f"    --output_dir checkpoints/diffusion_lidar \\\n"
        f"    --log_dir logs/diffusion_lidar"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
