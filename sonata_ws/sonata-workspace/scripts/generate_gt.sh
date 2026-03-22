#!/bin/bash
# Generate ground truth maps from SemanticKITTI sequences
# Run from project root: bash scripts/generate_gt.sh

set -e

# Default paths - adjust for your setup
SEQ_PATH="${1:-Datasets/SemanticKITTI/dataset/sequences}"
OUTPUT_DIR="${2:-Datasets/SemanticKITTI/dataset}"
VOXEL_SIZE="${3:-0.1}"

echo "Generating ground truth maps..."
echo "  Sequences: $SEQ_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Voxel size: $VOXEL_SIZE"

python data/map_from_scans.py \
  --path "$SEQ_PATH" \
  --output "$OUTPUT_DIR" \
  --voxel_size "$VOXEL_SIZE" \
  --sequences 00 01 02 03 04 05 06 07 08 09 10

echo ""
echo "Done. Use use_ground_truth_maps=True in SemanticKITTI dataset."
