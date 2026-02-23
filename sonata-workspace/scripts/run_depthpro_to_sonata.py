#!/usr/bin/env python3
"""
End-to-end pipeline:

RGB (SemanticKITTI) --Depth Pro--> point clouds --+ labels from voxels
                                                 |
                               build dataset ----+--> Sonata-LiDiff training

Runs:
  1) VoxFormerDepthPro steps 1–4
  2) Builds a dedicated dataset root for Depth Pro
  3) Generates ground-truth maps with map_from_scans (GPU backend if available)
  4) Trains Sonata-LiDiff (diffusion + refinement) on that dataset

Run from project root:
  cd ~/Simon_ws/sonata-workspace
  python scripts/run_depthpro_to_sonata.py
"""

import os
import subprocess
from typing import List


def run(cmd: List[str], cwd: str) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths (match paths_config defaults)
    workspace_dataset = os.path.expanduser("~/Simon_ws/dataset")
    kitti_root = os.path.join(workspace_dataset, "SemanticKITTI")
    vox_out_root = os.path.join(workspace_dataset, "VoxFormerDepthPro_out")
    sonata_dp_root = os.path.join(workspace_dataset, "sonata_depth_pro")

    sequences = [f"{i:02d}" for i in range(11)]  # 00–10

    print("=== 1) VoxFormerDepthPro: label preprocessing (voxels) ===")
    run(
        ["python", "VoxFormerDepthPro/scripts/1_prepare_labels.py"],
        cwd=repo_root,
    )

    print("\n=== 2) VoxFormerDepthPro: Depth Pro on RGB (per sequence) ===")
    for seq in sequences:
        run(
            [
                "python",
                "VoxFormerDepthPro/scripts/2_run_depth_pro.py",
                "--seq",
                seq,
                "--device",
                "cuda",
            ],
            cwd=repo_root,
        )

    print("\n=== 3) VoxFormerDepthPro: depth -> point clouds (.bin) ===")
    run(
        [
            "python",
            "VoxFormerDepthPro/scripts/3_depth_to_pointcloud.py",
            "--sequences",
            *sequences,
        ],
        cwd=repo_root,
    )

    print("\n=== 4) VoxFormerDepthPro: assign labels from voxels ===")
    run(
        [
            "python",
            "VoxFormerDepthPro/scripts/4_assign_labels_from_voxels.py",
            "--sequences",
            *sequences,
        ],
        cwd=repo_root,
    )

    print("\n=== 5) Build Sonata dataset root for Depth Pro ===")
    vox_lidar_pro = os.path.join(vox_out_root, "lidar_pro")
    vox_lidar_labeled = os.path.join(vox_out_root, "lidar_pro_labeled", "labels")
    kitti_dataset_root = os.path.join(kitti_root, "dataset")

    for seq in sequences:
        seq_dir = os.path.join(sonata_dp_root, "sequences", seq)
        velodyne_dir = os.path.join(seq_dir, "velodyne")
        labels_dir = os.path.join(seq_dir, "labels")
        os.makedirs(velodyne_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Point clouds from Depth Pro
        src_pc_dir = os.path.join(vox_lidar_pro, "sequences", seq)
        if os.path.isdir(src_pc_dir):
            for fname in sorted(os.listdir(src_pc_dir)):
                if fname.endswith(".bin"):
                    src = os.path.join(src_pc_dir, fname)
                    dst = os.path.join(velodyne_dir, fname)
                    if not os.path.exists(dst):
                        os.link(src, dst) if hasattr(os, "link") else None
                        if not os.path.exists(dst):
                            # Fallback to copy if hardlink failed
                            from shutil import copy2

                            copy2(src, dst)

        # Labels from voxels
        src_lbl_dir = os.path.join(vox_lidar_labeled, seq)
        if os.path.isdir(src_lbl_dir):
            for fname in sorted(os.listdir(src_lbl_dir)):
                if fname.endswith(".label"):
                    src = os.path.join(src_lbl_dir, fname)
                    dst = os.path.join(labels_dir, fname)
                    if not os.path.exists(dst):
                        if hasattr(os, "link"):
                            try:
                                os.link(src, dst)
                            except OSError:
                                from shutil import copy2

                                copy2(src, dst)
                        else:
                            from shutil import copy2

                            copy2(src, dst)

        # Poses + calib from SemanticKITTI
        src_seq_kitti = os.path.join(kitti_dataset_root, "sequences", seq)
        for name in ("poses.txt", "calib.txt"):
            src = os.path.join(src_seq_kitti, name)
            if os.path.exists(src):
                dst = os.path.join(seq_dir, name)
                if not os.path.exists(dst):
                    from shutil import copy2

                    copy2(src, dst)

    print("Depth Pro dataset root:", sonata_dp_root)

    print("\n=== 6) Generate ground truth maps for Depth Pro dataset ===")
    seq_path = os.path.join(sonata_dp_root, "sequences")
    run(
        [
            "python",
            "data/map_from_scans.py",
            "--path",
            seq_path,
            "--output",
            sonata_dp_root,
            "--voxel_size",
            "0.1",
            "--backend",
            "torch",
            "--sequences",
            *sequences,
        ],
        cwd=repo_root,
    )

    print("\n=== 7) Train Sonata-LiDiff (diffusion) on Depth Pro dataset ===")
    run(
        [
            "python",
            "training/train_diffusion.py",
            "--data_path",
            sonata_dp_root,
            "--output_dir",
            "checkpoints/diffusion_depthpro",
            "--log_dir",
            "logs/diffusion_depthpro",
        ],
        cwd=repo_root,
    )

    print("\n=== 8) Train refinement network on Depth Pro dataset ===")
    run(
        [
            "python",
            "training/train_refinement.py",
            "--data_path",
            sonata_dp_root,
            "--output_dir",
            "checkpoints/refinement_depthpro",
            "--log_dir",
            "logs/refinement_depthpro",
        ],
        cwd=repo_root,
    )

    print("\nAll done. Depth Pro → Sonata-LiDiff pipeline finished.")


if __name__ == "__main__":
    main()

