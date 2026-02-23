#!/usr/bin/env python3
"""
Assign per-point labels to Depth Pro point clouds using voxel label lookup.

Points and voxels are in the same sensor frame per frame, so no poses needed.
Requires: SemanticKITTI voxels/, Depth Pro point clouds (.bin), matching frame IDs.
"""

import os
import sys
import argparse
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.voxel_to_point_labels import process_frame
from paths_config import DEFAULT_KITTI_ROOT, get_lidar_pro_root, get_lidar_pro_labeled_root


def run_sequence(
    pointcloud_dir: str,
    voxel_dir: str,
    output_labels_dir: str,
) -> int:
    """Process one sequence. Returns number of frames processed."""
    os.makedirs(output_labels_dir, exist_ok=True)
    bin_files = sorted(f for f in os.listdir(pointcloud_dir) if f.endswith(".bin"))
    total_labeled = 0
    for fn in tqdm(bin_files, desc=os.path.basename(pointcloud_dir)):
        frame_id = fn[:-4]
        label_fn = frame_id + ".label"
        voxel_path = os.path.join(voxel_dir, label_fn)
        if not os.path.exists(voxel_path):
            continue
        points_path = os.path.join(pointcloud_dir, fn)
        out_path = os.path.join(output_labels_dir, label_fn)
        n = process_frame(points_path, voxel_path, out_path)
        total_labeled += n
    return len(bin_files)


def main():
    parser = argparse.ArgumentParser(
        description="Assign voxel labels to Depth Pro point clouds"
    )
    parser.add_argument(
        "--pointcloud_root", "-p", type=str, default=get_lidar_pro_root(),
        help="Root with sequences/XX/ containing .bin point clouds (default: from paths_config)",
    )
    parser.add_argument(
        "--voxel_root", "-v", type=str, default=DEFAULT_KITTI_ROOT,
        help="SemanticKITTI root with dataset/sequences/XX/voxels/ (default: from paths_config)",
    )
    parser.add_argument(
        "--output_root", "-o", type=str, default=get_lidar_pro_labeled_root(),
        help="Output root for labels/XX/*.label (default: from paths_config)",
    )
    parser.add_argument(
        "--sequences", "-s", type=str, nargs="+", default=None,
        help="Sequence IDs (default: 00-10)",
    )
    args = parser.parse_args()

    sequences = args.sequences or ["%02d" % i for i in range(11)]
    for seq in sequences:
        pc_dir = os.path.join(args.pointcloud_root, "sequences", seq)
        vox_dir = os.path.join(args.voxel_root, "dataset", "sequences", seq, "voxels")
        out_dir = os.path.join(args.output_root, "labels", seq)
        if not os.path.exists(pc_dir) or not os.path.exists(vox_dir):
            print("Skipping %s: missing pc_dir or voxel_dir" % seq)
            continue
        n = run_sequence(pc_dir, vox_dir, out_dir)
        print("Sequence %s: %d frames" % (seq, n))


if __name__ == "__main__":
    main()
