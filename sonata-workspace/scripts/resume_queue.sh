#!/usr/bin/env bash
# Resume runs killed on Jul 8: adv_only_1e-4 (died at ep0), token critic (died at start)
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/sonata_ws/sonata-workspace-fixed/sonata-workspace

echo "[resume] 1/2 adv_only_1e-4 restart $(date)"
python run_vae_gan_ablations.py --only adv_only_1e-4 --execute \
  --gt_subdir ground_truth_v1 --device cuda
echo "[resume] adv_only_1e-4 exit=$? $(date)"

echo "[resume] 2/2 token critic restart $(date)"
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
echo "[resume] token critic exit=$? $(date)"
echo "[resume] ALL DONE $(date)"
