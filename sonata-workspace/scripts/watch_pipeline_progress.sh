#!/bin/bash
# Смотреть прогресс пайплайна Depth Pro в реальном времени.
# Запуск: ./scripts/watch_pipeline_progress.sh

OUT="${OUT:-/workspace/dataset}"
DEPTH="$OUT/VoxFormerDepthPro/depth/sequences"
LIDAR="$OUT/VoxFormerDepthPro/lidar_pro/sequences"
NEED_00=4541
NEED_01=1101

while true; do
  clear
  echo "=== Depth Pro pipeline progress ($(date '+%H:%M:%S')) ==="
  echo ""

  # Какой процесс крутится
  if pgrep -f "3_depth_to_pointcloud" >/dev/null; then
    echo "Current step: 3 (depth -> point clouds)"
  elif pgrep -f "4_assign_labels" >/dev/null; then
    echo "Current step: 4 (assign labels from voxels)"
  elif pgrep -f "2_run_depth_pro" >/dev/null; then
    echo "Current step: 2 (Depth Pro on RGB)"
  elif pgrep -f "map_from_scans" >/dev/null; then
    echo "Current step: 6 or LiDAR map_from_scans"
  elif pgrep -f "run_depthpro_to_sonata" >/dev/null; then
    echo "Current step: 5 or 7-8 (build dataset / train)"
  else
    echo "Current step: (no pipeline process found)"
  fi
  echo ""

  LABELED="$OUT/VoxFormerDepthPro/lidar_pro_labeled/labels"

  # Step 3: прогресс по seq 00 и 01 (.bin)
  if [ -d "$LIDAR/00" ]; then
    n00=$(find "$LIDAR/00" -maxdepth 1 -name "*.bin" 2>/dev/null | wc -l)
    pct00=$((n00 * 100 / NEED_00))
    echo "  [step 3] seq 00: $n00 / $NEED_00 .bin  ($pct00%)"
  fi
  if [ -d "$LIDAR/01" ]; then
    n01=$(find "$LIDAR/01" -maxdepth 1 -name "*.bin" 2>/dev/null | wc -l)
    pct01=$((n01 * 100 / NEED_01))
    echo "  [step 3] seq 01: $n01 / $NEED_01 .bin  ($pct01%)"
  fi

  # Step 4: прогресс по seq 00 и 01 (.label)
  if [ -d "$LABELED/00" ]; then
    l00=$(find "$LABELED/00" -maxdepth 1 -name "*.label" 2>/dev/null | wc -l)
    pct00=$((l00 * 100 / NEED_00))
    echo "  [step 4] seq 00: $l00 / $NEED_00 .label ($pct00%)"
  fi
  if [ -d "$LABELED/01" ]; then
    l01=$(find "$LABELED/01" -maxdepth 1 -name "*.label" 2>/dev/null | wc -l)
    pct01=$((l01 * 100 / NEED_01))
    echo "  [step 4] seq 01: $l01 / $NEED_01 .label ($pct01%)"
  fi

  echo ""
  echo "Update every 10s. Ctrl+C to exit."
  sleep 10
done
