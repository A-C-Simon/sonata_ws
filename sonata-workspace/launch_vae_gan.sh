#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace

TIMESTAMP=
LOG_DIR=$PWD/logs/vae_gan_20260523_1217
OUT_DIR=$PWD/checkpoints/vae_gan_v1gt_20260523_1217
mkdir -p $LOG_DIR $OUT_DIR

echo "=== VAE-GAN training launch 20260523_1217 ===" | tee $LOG_DIR/run.log
echo "GT: v1 (ground_truth_v1)" | tee -a $LOG_DIR/run.log
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a $LOG_DIR/run.log
echo "" | tee -a $LOG_DIR/run.log

python training/train_vae_gan.py \
  --resume_vae checkpoints/point_vae_v3/best_point_vae.pth \
  --num_epochs 30 \
  --n_critic 2 \
  --batch_size 8 \
  --lr_gen 1e-4 \
  --lambda_fm 1.0 \
  --ema_start_epoch 10 \
  --seed 42 \
  --gt_subdir ground_truth_v1 \
  --output_dir $OUT_DIR \
  --log_dir $LOG_DIR \
  2>&1 | tee -a $LOG_DIR/run.log

EXIT=${PIPESTATUS[0]}
echo "" | tee -a $LOG_DIR/run.log
echo "=== TRAINING ENDED (exit=$EXIT) ===" | tee -a $LOG_DIR/run.log
ssh -o ConnectTimeout=5 business 'telegram-notify "vae_gan training done exit= seed=42 v1gt 30ep "' || echo notify-failed
