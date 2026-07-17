#!/usr/bin/env bash
# Stage 4: TokenCritic head-to-head — waits for main queue to finish first
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/sonata_ws/sonata-workspace-fixed/sonata-workspace

echo "[stage4] waiting for main queue to finish..."
while pgrep -f vae_gan_queue.sh > /dev/null; do sleep 300; done
echo "[stage4] main queue done, starting token critic at $(date)"

python training/train_vae_gan.py \
  --resume_vae checkpoints/point_vae_v3/best_point_vae.pth \
  --data_path $HOME/Simon_ws/dataset/SemanticKITTI/dataset \
  --batch_size 4 --seed 42 --gt_subdir ground_truth_v1 --device cuda \
  --output_dir checkpoints/vae_gan_token_cond_adaptive_r1 \
  --log_dir logs/vae_gan_token_cond_adaptive_r1 \
  --num_epochs 25 --disc_warmup_epochs 2 --lambda_adv_ramp_epochs 3 \
  --lambda_adv 0.01 --lambda_fm 0.0 \
  --critic token --critic_tokens 32 --critic_dim 256 --conditional_critic \
  --adaptive_adv --adaptive_max 1e4 --gp_mode r1 --log_grad_align
echo "[stage4] token critic exit=$? $(date)"
