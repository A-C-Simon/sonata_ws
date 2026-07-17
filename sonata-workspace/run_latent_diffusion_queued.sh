#!/usr/bin/env bash
# Queued latent-diffusion (V3 pivot) run. Waits for Chidera's VAE-GAN to finish
# and the GPU to free, then launches. Does NOT contend with his run.
# Our run = fix #4 (Sonata-conditioned latent diffusion on the frozen V3 32x1024 latent),
# complementary + orthogonal to his VAE-GAN. Eval'd separately via evaluate_latent_diffusion.py.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0

CHIDERA_PID=3948773
echo "[ld-queue] $(date) waiting for Chidera VAE-GAN PID $CHIDERA_PID ..."
while kill -0 "$CHIDERA_PID" 2>/dev/null; do sleep 60; done
echo "[ld-queue] $(date) Chidera process exited. Waiting for GPU memory to release ..."
# wait until <2GB used so his process has fully freed VRAM (and nobody else grabbed it)
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-9999}" -lt 2000 ] && break
  sleep 30
done
echo "[ld-queue] $(date) GPU free (${USED} MiB used). Launching latent diffusion (V3 pivot)."

OUT=checkpoints/latent_diffusion_v3pivot
LOG=logs/latent_diffusion_v3pivot
mkdir -p "$LOG"
echo "TRAINING IN PROGRESS - latent_diffusion_v3pivot - started $(date)" > ~/DO_NOT_REBOOT_TRAINING.txt

python seed_then_train.py \
  --vae_ckpt checkpoints/point_vae_v3/best_point_vae.pth \
  --data_path /home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset \
  --freeze_encoder \
  --num_latent_tokens 32 --num_cond_tokens 32 \
  --hidden_dim 1024 --num_denoiser_blocks 8 --num_heads 4 \
  --num_timesteps 1000 --schedule cosine --denoising_steps 50 \
  --point_max_complete 8000 --point_max_partial 20000 \
  --batch_size 4 --num_epochs 60 --learning_rate 1e-4 --gradient_clip 1.0 \
  --eval_freq 1 --save_freq 5 \
  --output_dir "$OUT" --log_dir "$LOG" \
  > "$LOG/run.out" 2>&1
RC=$?
echo "[ld-queue] $(date) latent diffusion exited (rc=$RC). Checkpoints in $OUT, log $LOG/run.out"
echo "latent_diffusion_v3pivot FINISHED $(date) rc=$RC" >> ~/DO_NOT_REBOOT_TRAINING.txt
