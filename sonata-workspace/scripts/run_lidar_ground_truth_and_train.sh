#!/bin/bash
# 1) map_from_scans по LiDAR (SemanticKITTI velodyne)
# 2) Обучение Sonata-LiDiff с корнем SemanticKITTI/dataset
# Запускать из корня репозитория. Не ждёт Depth Pro (шаги 2–3–4).

set -e
cd "$(dirname "$0")/.."
dataset="${dataset:-/workspace/dataset}"
DATA="$dataset"
KITTI_SEQ="$DATA/SemanticKITTI/dataset/sequences"
KITTI_ROOT="$DATA/SemanticKITTI/dataset"

echo "=== 1) Generate ground truth from LiDAR (map_from_scans) ==="
python data/map_from_scans.py \
  -p "$KITTI_SEQ" \
  -o "$KITTI_ROOT" \
  -v 0.1 -b torch \
  -s 00 01 02 03 04 05 06 07 08 09 10

echo "=== 2) Train diffusion (LiDAR root) ==="
python training/train_diffusion.py \
  --data_path "$KITTI_ROOT" \
  --output_dir checkpoints/diffusion_lidar \
  --log_dir logs/diffusion_lidar

echo "=== 3) Train refinement (LiDAR root) ==="
python training/train_refinement.py \
  --data_path "$KITTI_ROOT" \
  --output_dir checkpoints/refinement_lidar \
  --log_dir logs/refinement_lidar

echo "Done. LiDAR-based Sonata-LiDiff checkpoints in checkpoints/diffusion_lidar and checkpoints/refinement_lidar."
