#!/usr/bin/env bash
# VAE-GAN diagnosis queue — launched by Claude Code 2026-07-06
# Priority: 1) recon ceiling  2) balanced GAN (adaptive+R1)  3) missing ablation arms
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/sonata_ws/sonata-workspace-fixed/sonata-workspace

echo "[queue] waiting for running eval to finish..."
while pgrep -f evaluate_vae_gan.py > /dev/null; do sleep 60; done
echo "[queue] eval done, starting runs at $(date)"

COMMON="--resume_vae checkpoints/point_vae_v3/best_point_vae.pth \
  --data_path $HOME/Simon_ws/dataset/SemanticKITTI/dataset \
  --batch_size 4 --seed 42 --gt_subdir ground_truth_v1 --device cuda"

# ---- 1. Recon ceiling: 40 epochs pure VAE fine-tune, critic disabled ----
echo "[queue] 1/3 recon_long_40ep $(date)"
python vae_gan/train_vae_gan.py $COMMON \
  --output_dir checkpoints/vae_gan_recon_long_40ep \
  --log_dir logs/vae_gan_recon_long_40ep \
  --num_epochs 40 --disc_warmup_epochs 999 \
  --lambda_adv 0.0 --lambda_fm 0.0
echo "[queue] recon_long_40ep exit=$? $(date)"

# ---- 2. Balanced GAN: adaptive adversarial weight + R1 penalty ----
echo "[queue] 2/3 adv_adaptive_r1 $(date)"
python vae_gan/train_vae_gan.py $COMMON \
  --output_dir checkpoints/vae_gan_adv_adaptive_r1 \
  --log_dir logs/vae_gan_adv_adaptive_r1 \
  --num_epochs 25 --disc_warmup_epochs 2 --lambda_adv_ramp_epochs 3 \
  --lambda_adv 0.01 --lambda_fm 0.0 \
  --adaptive_adv --adaptive_max 1e4 --gp_mode r1 --log_grad_align
echo "[queue] adv_adaptive_r1 exit=$? $(date)"

# ---- 3. Missing ablation arms (same roots as existing recon_only control) ----
echo "[queue] 3/3 ablation sweep $(date)"
python vae_gan/run_vae_gan_ablations.py \
  --only fm_only,adv_only_1e-4,adv_only_1e-3 --execute \
  --gt_subdir ground_truth_v1 --device cuda
echo "[queue] sweep exit=$? $(date)"
echo "[queue] ALL DONE $(date)"
