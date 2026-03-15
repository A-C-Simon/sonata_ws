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

Run from project root (set env for your paths):
  export dataset=/workspace/dataset OUT=/workspace/dataset
  cd /workspace/sonata-workspace
  python scripts/run_depthpro_to_sonata.py
"""

import os
import sys
import subprocess
from typing import List


def run(cmd: List[str], cwd: str) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from VoxFormerDepthPro.paths_config import (
        DEFAULT_KITTI_ROOT,
        DEFAULT_VOXFORMER_OUT,
        OUT_ROOT,
        get_preprocess_root,
    )

    # KITTI = $dataset/SemanticKITTI, VoxFormer out = $OUT/VoxFormerDepthPro
    kitti_root = DEFAULT_KITTI_ROOT
    vox_out_root = DEFAULT_VOXFORMER_OUT
    sonata_dp_root = os.path.join(OUT_ROOT, "sonata_depth_pro")
    # Depth Pro loads ./checkpoints/depth_pro.pt relative to cwd; use ml-depth-pro root
    depth_pro_root = os.environ.get("DEPTH_PRO_ROOT", "/workspace/ml-depth-pro")

    sequences = [f"{i:02d}" for i in range(11)]  # 00–10

    # Step 1: skip if preprocess output already exists for all sequences
    preprocess_root = get_preprocess_root()
    def preprocess_done():
        for seq in sequences:
            lbl_dir = os.path.join(preprocess_root, "labels", seq)
            if not os.path.isdir(lbl_dir):
                return False
            if not any(f.endswith(".npy") for f in os.listdir(lbl_dir)):
                return False
        return True

    # Step 2: depth/sequences/XX/*.npy — skip only if count matches image_2 (avoid partial runs)
    depth_root = os.path.join(vox_out_root, "depth", "sequences")
    image2_root = os.path.join(kitti_root, "dataset", "sequences")
    def depth_done_for(seq):
        depth_dir = os.path.join(depth_root, seq)
        image_dir = os.path.join(image2_root, seq, "image_2")
        if not os.path.isdir(depth_dir):
            return False
        n_depth = len([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
        if n_depth == 0:
            return False
        if not os.path.isdir(image_dir):
            return True  # no source images, consider done
        n_images = len([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        return n_depth >= n_images

    # Step 3: lidar_pro/sequences/XX/*.bin
    lidar_pro_root = os.path.join(vox_out_root, "lidar_pro", "sequences")
    def lidar_pro_done():
        for seq in sequences:
            d = os.path.join(lidar_pro_root, seq)
            if not os.path.isdir(d) or not any(f.endswith(".bin") for f in os.listdir(d)):
                return False
        return True

    # Step 4: lidar_pro_labeled/labels/XX/*.label
    lidar_labeled_root = os.path.join(vox_out_root, "lidar_pro_labeled", "labels")
    def lidar_labeled_done():
        for seq in sequences:
            d = os.path.join(lidar_labeled_root, seq)
            if not os.path.isdir(d) or not any(f.endswith(".label") for f in os.listdir(d)):
                return False
        return True

    print("=== 1) VoxFormerDepthPro: label preprocessing (voxels) ===")
    if preprocess_done():
        print("(skip: preprocess/labels/00..10 already present)")
    else:
        run(
            ["python", "VoxFormerDepthPro/scripts/1_prepare_labels.py"],
            cwd=repo_root,
        )

    print("\n=== 2) VoxFormerDepthPro: Depth Pro on RGB (per sequence) ===")
    script_2 = os.path.join(repo_root, "VoxFormerDepthPro", "scripts", "2_run_depth_pro.py")
    for seq in sequences:
        if depth_done_for(seq):
            print(f"(skip seq {seq}: depth already present)")
            continue
        run(
            [
                sys.executable,
                script_2,
                "--seq",
                seq,
                "--device",
                "cuda",
            ],
            cwd=depth_pro_root,  # so depth_pro finds ./checkpoints/depth_pro.pt
        )

    print("\n=== 3) VoxFormerDepthPro: depth -> point clouds (.bin) ===")
    # Always run step 3: script 3_depth_to_pointcloud.py skips per-seq when n_bin >= n_depth.
    # So after re-running step 2 only for 00/01, step 3 will regenerate 00/01 and skip 02-10.
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
    # Always run step 4: script skips or overwrites per sequence; needed after step 3 updated 00/01.
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

