#!/usr/bin/env python3
"""Convert depth maps (.npy) to LiDAR-style point clouds (.bin)."""

import sys
import os
import argparse
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.depth_to_pointcloud import process_sequence
from paths_config import get_dataset_root, get_depth_root, get_lidar_pro_root


def run_sequences(depth_root, calib_root, save_root, sequences=None, max_depth=80.0):
    """Process multiple sequences. Skips seq that already have .bin (resume after failure)."""
    if sequences is None:
        sequences = ["%02d" % i for i in range(22)]
    for seq in sequences:
        depth_dir = os.path.join(depth_root, "sequences", seq)
        calib_dir = os.path.join(calib_root, "sequences", seq)
        save_dir = os.path.join(save_root, "sequences", seq)
        if not os.path.isdir(depth_dir) or not os.path.isdir(calib_dir):
            continue
        # Skip if this seq already has .bin (e.g. after re-run following step 3 failure)
        if os.path.isdir(save_dir) and any(f.endswith(".bin") for f in os.listdir(save_dir)):
            n_depth = len([f for f in os.listdir(depth_dir) if f.endswith(".npy") and "std" not in f])
            n_bin = len([f for f in os.listdir(save_dir) if f.endswith(".bin")])
            if n_bin >= n_depth:
                print("Seq %s: skip (already %d .bin)" % (seq, n_bin))
                continue
        n = process_sequence(depth_dir, calib_dir, save_dir, max_depth=max_depth)
        print("Seq %s: %d clouds" % (seq, n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_root", type=str, default=get_depth_root(),
                        help="Root with depth/sequences/XX/*.npy (default: from paths_config)")
    parser.add_argument("--calib_root", type=str, default=get_dataset_root(),
                        help="Dataset root with sequences/XX/calib.txt (default: from paths_config)")
    parser.add_argument("--save_root", type=str, default=get_lidar_pro_root(),
                        help="Output root for sequences/XX/*.bin (default: from paths_config)")
    parser.add_argument("--sequences", type=str, default=None, nargs="+",
                        help="e.g. 00 01 02 (default: all 00-21)")
    parser.add_argument("--max_depth", type=float, default=80.0)
    args = parser.parse_args()
    run_sequences(
        args.depth_root,
        args.calib_root,
        args.save_root,
        sequences=args.sequences,
        max_depth=args.max_depth,
    )
