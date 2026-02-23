#!/bin/bash
# Example: run full pipeline. Set KITTI_ROOT and OUT_ROOT.
set -e
KITTI_ROOT=${KITTI_ROOT:-/path/to/semantic_kitti}
OUT_ROOT=${OUT_ROOT:-./preprocess}
CALIB_ROOT=${CALIB_ROOT:-$KITTI_ROOT/data_odometry_calib}

python scripts/1_prepare_labels.py -r "$KITTI_ROOT" -p "$OUT_ROOT"

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  python scripts/2_run_depth_pro.py \
    --image_dir "$KITTI_ROOT/dataset/sequences/$seq/image_2" \
    --depth_dir "$OUT_ROOT/depth/sequences/$seq"
done

python scripts/3_depth_to_pointcloud.py \
  --depth_root "$OUT_ROOT/depth" \
  --calib_root "$CALIB_ROOT" \
  --save_root "$OUT_ROOT/lidar_pro"
