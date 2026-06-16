#!/usr/bin/env bash
# Wait for the recon_only control to finish, then launch the primary fix run:
# token cross-attention critic + VQGAN-adaptive adversarial weight + R1 penalty
# + conditional (on partial scan), FM off. Warm-started from V3 on v2 GT.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
export PYTORCH_ALLOC_CONF=expandable_segments:True

CONTROL_PID=2619771
echo "[queue] waiting for control PID $CONTROL_PID ... $(date)"
while kill -0 "$CONTROL_PID" 2>/dev/null; do sleep 60; done
echo "[queue] control finished, starting fix at $(date)"

OUT=checkpoints/vae_gan_fix_v2/token_adapt_r1_cond
LOG=logs/vae_gan_fix_v2/token_adapt_r1_cond
mkdir -p "$LOG"

python training/train_vae_gan.py \
  --resume_vae checkpoints/point_vae_v3/best_point_vae.pth \
  --data_path /home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset \
  --gt_subdir ground_truth \
  --critic token --adaptive_adv --gp_mode r1 --conditional_critic \
  --lambda_adv 1.0 --lambda_fm 0 --n_critic 3 \
  --num_epochs 20 --disc_warmup_epochs 2 --lambda_adv_ramp_epochs 3 \
  --batch_size 4 --seed 42 --log_grad_align \
  --output_dir "$OUT" --log_dir "$LOG" \
  > "$LOG/run.out" 2>&1

echo "[queue] fix finished at $(date) (exit $?)"
