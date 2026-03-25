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
import argparse
from typing import List


def run(cmd: List[str], cwd: str) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_gt_exists(data_root: str, gt_subdir: str, sequences: List[str], n_check: int = 3) -> None:
    """
    Verify a few GT files exist for requested sequences.
    Expects: <data_root>/<gt_subdir>/<seq>/<frame>.npz
    """
    import glob

    for seq in sequences:
        gt_seq = os.path.join(data_root, gt_subdir, seq)
        if not os.path.isdir(gt_seq):
            raise FileNotFoundError(f"GT folder not found: {gt_seq}")
        files = sorted(glob.glob(os.path.join(gt_seq, "*.npz")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npz GT files found in: {gt_seq}")
        print(f"[GT check] {gt_seq}: found {len(files)} .npz files")
        for p in files[:n_check]:
            print(f"  - {os.path.basename(p)}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="End-to-end LiDAR→GT→Sonata-LiDiff pipeline (diffusion + refinement)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"),
        help="Dataset root containing sequences/ and ground-truth folder(s)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequence IDs to use (default: 00-10)",
    )
    parser.add_argument(
        "--gt_variant",
        type=str,
        default="ground_truth",
        help="Ground-truth subdir to USE for training (e.g. ground_truth_v2)",
    )
    parser.add_argument(
        "--generate_gt",
        action="store_true",
        help="Generate GT maps before training (writes to --gt_variant)",
    )
    parser.add_argument(
        "--force_generate_gt",
        action="store_true",
        help="Re-generate GT even if files already exist (only supported for gt_variant=ground_truth)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel size for GT generation (map_from_scans)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["numpy", "open3d", "torch"],
        help="Voxelization backend for GT generation",
    )
    parser.add_argument(
        "--max_scans_per_sequence",
        type=int,
        default=None,
        help="Cap frames per sequence for a minimal working run",
    )
    parser.add_argument(
        "--diffusion_epochs",
        type=int,
        default=None,
        help="Override diffusion training epochs (passes --num_epochs to training script)",
    )
    parser.add_argument(
        "--diffusion_batch_size",
        type=int,
        default=1,
        help="Diffusion batch size (lower to avoid OOM)",
    )
    parser.add_argument(
        "--diffusion_max_points",
        type=int,
        default=4000,
        help="Diffusion max_points passed to dataset (lower to avoid OOM)",
    )
    parser.add_argument(
        "--refinement_epochs",
        type=int,
        default=None,
        help="Override refinement training epochs (passes --num_epochs to training script)",
    )
    parser.add_argument(
        "--encoder_backend",
        type=str,
        default=None,
        choices=["sonata", "dummy"],
        help="Pass-through to train_diffusion_2.py (use dummy for smoke-test)",
    )
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Enable dataset debug prints/checks in training scripts",
    )
    args = parser.parse_args()

    kitti_dataset_root = os.path.expanduser(args.data_path)
    sequences = args.sequences if args.sequences is not None else [f"{i:02d}" for i in range(11)]  # 00–10

    if args.generate_gt:
        if args.gt_variant != "ground_truth":
            raise ValueError(
                "GT generation in this workspace (data/map_from_scans.py) writes only to "
                "'ground_truth/'. To train on an existing folder like 'ground_truth_v2', "
                "omit --generate_gt and set --gt_variant ground_truth_v2."
            )
        print("=== 1) Generate ground truth maps from LiDAR (SemanticKITTI) ===")
        seq_path = os.path.join(kitti_dataset_root, "sequences")
        # Skip re-generation if files already exist (unless forced)
        gt_seq_dir = os.path.join(kitti_dataset_root, args.gt_variant, sequences[0])
        if (not args.force_generate_gt) and os.path.isdir(gt_seq_dir) and len(os.listdir(gt_seq_dir)) > 0:
            print(f"GT already exists under {os.path.join(kitti_dataset_root, args.gt_variant)}. Skipping (use --force_generate_gt to overwrite).")
        else:
            run(
                [
                    "python",
                    "data/map_from_scans.py",
                    "--path",
                    seq_path,
                    "--output",
                    kitti_dataset_root,
                    "--voxel_size",
                    str(args.voxel_size),
                    "--backend",
                    args.backend,
                    "--sequences",
                    *sequences,
                ],
                cwd=repo_root,
            )
    else:
        print("=== 1) Skipping GT generation (using existing GT) ===")
        _ensure_gt_exists(kitti_dataset_root, args.gt_variant, sequences)

    print("\n=== 2) Train Sonata-LiDiff (diffusion) on LiDAR dataset ===")
    diff_cmd = [
        "python",
        "training/train_diffusion_2.py",
        "--data_path",
        kitti_dataset_root,
        "--gt_variant",
        args.gt_variant,
        "--output_dir",
        "checkpoints/diffusion_lidar",
        "--log_dir",
        "logs/diffusion_lidar",
        "--sequences",
        *sequences,
        "--batch_size",
        str(args.diffusion_batch_size),
        "--max_points",
        str(args.diffusion_max_points),
    ]
    if args.max_scans_per_sequence is not None:
        diff_cmd += ["--max_scans_per_sequence", str(args.max_scans_per_sequence)]
    if args.diffusion_epochs is not None:
        diff_cmd += ["--num_epochs", str(args.diffusion_epochs)]
    if args.encoder_backend is not None:
        diff_cmd += ["--encoder_backend", args.encoder_backend]
    if args.debug_data:
        diff_cmd += ["--debug_data"]
    run(
        diff_cmd,
        cwd=repo_root,
    )

    print("\n=== 3) Train refinement network on LiDAR dataset ===")
    ref_cmd = [
        "python",
        "training/train_refinement_2.py",
        "--data_path",
        kitti_dataset_root,
        "--gt_variant",
        args.gt_variant,
        "--output_dir",
        "checkpoints/refinement_lidar",
        "--log_dir",
        "logs/refinement_lidar",
        "--sequences",
        *sequences,
    ]
    if args.max_scans_per_sequence is not None:
        ref_cmd += ["--max_scans_per_sequence", str(args.max_scans_per_sequence)]
    if args.refinement_epochs is not None:
        ref_cmd += ["--num_epochs", str(args.refinement_epochs)]
    if args.debug_data:
        ref_cmd += ["--debug_data"]
    run(
        ref_cmd,
        cwd=repo_root,
    )

    print("\nAll done. LiDAR-only Sonata-LiDiff pipeline finished.")


if __name__ == "__main__":
    main()

