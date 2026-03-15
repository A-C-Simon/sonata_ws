#!/usr/bin/env python3
"""
End-to-end pipeline for LiDAR-only SemanticKITTI (вариант 1: минимум места, один проход map_from_scans).

  1) map_from_scans --save_only_map_world  → ground_truth/XX/map_world.npz (один раз)
  2) precompute_voxelized_dataset --backend torch [--max_map_points 2M]  → voxelized_cache/XX/*.npz
  3) train_diffusion + train_refinement с --voxelized_cache_dir

Запуск из корня репозитория:
  python scripts/run_lidar_to_sonata.py
"""

import os
import subprocess
from typing import List


def run(cmd: List[str], cwd: str) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kitti_dataset_root = os.path.expanduser(
        "~/Simon_ws/dataset/SemanticKITTI/dataset"
    )
    sequences = [f"{i:02d}" for i in range(11)]  # 00–10

    print("=== 1) Generate ground truth maps from LiDAR (SemanticKITTI) ===")
    seq_path = os.path.join(kitti_dataset_root, "sequences")
    run(
        [
            "python",
            "data/map_from_scans.py",
            "--path",
            seq_path,
            "--output",
            kitti_dataset_root,
            "--voxel_size",
            "0.1",
            "--backend",
            "torch",
            "--save_only_map_world",
            "--sequences",
            *sequences,
        ],
        cwd=repo_root,
    )

    voxelized_cache = os.path.join(kitti_dataset_root, "voxelized_cache")
    print("\n=== 1.5) Precompute voxelized cache (map_world -> SК кадров, GPU) ===")
    run(
        [
            "python",
            "data/precompute_voxelized_dataset.py",
            "--data_path",
            kitti_dataset_root,
            "--output_dir",
            voxelized_cache,
            "--voxel_size",
            "0.05",
            "--backend",
            "torch",
            "--max_map_points",
            "2000000",
            "--sequences",
            *sequences,
        ],
        cwd=repo_root,
    )

    print("\n=== 2) Train Sonata-LiDiff (diffusion) on LiDAR dataset ===")
    run(
        [
            "python",
            "training/train_diffusion.py",
            "--data_path",
            kitti_dataset_root,
            "--voxelized_cache_dir",
            voxelized_cache,
            "--output_dir",
            "checkpoints/diffusion_lidar",
            "--log_dir",
            "logs/diffusion_lidar",
        ],
        cwd=repo_root,
    )

    print("\n=== 3) Train refinement network on LiDAR dataset ===")
    run(
        [
            "python",
            "training/train_refinement.py",
            "--data_path",
            kitti_dataset_root,
            "--output_dir",
            "checkpoints/refinement_lidar",
            "--log_dir",
            "logs/refinement_lidar",
        ],
        cwd=repo_root,
    )

    print("\nAll done. LiDAR-only Sonata-LiDiff pipeline finished.")


if __name__ == "__main__":
    main()

