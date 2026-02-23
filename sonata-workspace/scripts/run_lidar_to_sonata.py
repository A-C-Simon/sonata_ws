#!/usr/bin/env python3
"""
End-to-end pipeline for LiDAR-only SemanticKITTI:

  velodyne + labels + poses + calib
    -> ground_truth maps (map_from_scans)
    -> Sonata-LiDiff training (diffusion + refinement)

Run from project root:
  cd ~/Simon_ws/sonata-workspace
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

