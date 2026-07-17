#!/usr/bin/env bash
# Eval VAE-GAN against V3 baseline with patched 20k-subsample protocol.
# Usage: bash launch_eval.sh [CKPT_PATH]
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace

CKPT=${1:-checkpoints/vae_gan_v1gt_20260523_1217/best_vae_gan.pth}
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_DIR=evaluation_vae_gan_v1gt_${TIMESTAMP}

echo '=== VAE-GAN eval ==='
echo "ckpt: $CKPT"
echo "baseline V3: checkpoints/point_vae_v3/best_point_vae.pth"
echo "output: $OUT_DIR"
echo "protocol: 20k subsample (RA-L canonical), 50 frames, v1 GT"
echo ''

python evaluate_vae_gan.py \
  --ckpt "$CKPT" \
  --baseline_ckpt checkpoints/point_vae_v3/best_point_vae.pth \
  --output_dir "$OUT_DIR" \
  --num_samples 50 \
  --point_max_complete 20000 \
  --seed 42 \
  --gt_subdir ground_truth_v1 \
  --sequence 08

echo ''
echo '=== eval complete ==='
ssh -o ConnectTimeout=5 business 'telegram-notify "vae_gan eval done out=${OUT_DIR}"' 2>/dev/null || echo notify-skipped
