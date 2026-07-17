#!/usr/bin/env bash
# Fully autonomous queue on compute (single GPU). Waits for Chidera's VAE-GAN to finish,
# then runs OUR runs sequentially + auto-evaluates each + writes a SUMMARY. No supervision needed.
#   RUN 1 = OUR VAE-GAN (newest settings + our ema_start_epoch=2 fix, seed 43) -> the GOAL run
#   RUN 2 = latent-diffusion V3 pivot (fix #4)
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
WS=/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
cd "$WS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
DATA=/home/anywherevla/Simon_ws/dataset/SemanticKITTI/dataset
V3=checkpoints/point_vae_v3/best_point_vae.pth
SUM=/home/anywherevla/OVERNIGHT_SUMMARY.txt
say(){ echo "$(date) | $*" | tee -a "$SUM"; }

CHIDERA_PID=3948773
wait_gpu(){ while :; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${U:-9999}" -lt 2000 ] && break; sleep 30; done; }

: > "$SUM"
say "[queue] waiting for Chidera VAE-GAN PID $CHIDERA_PID to finish ..."
while kill -0 "$CHIDERA_PID" 2>/dev/null; do sleep 60; done
say "[queue] Chidera done. waiting for GPU to free ..."; wait_gpu; say "[queue] GPU free."

# ===================== RUN 1: OUR VAE-GAN (the goal) =====================
OUT1=checkpoints/vae_gan_fix_v2/ours_ema2_seed43; LOG1=logs/vae_gan_fix_v2/ours_ema2_seed43; mkdir -p "$LOG1"
echo "TRAINING IN PROGRESS - run1 our VAE-GAN - $(date)" > ~/DO_NOT_REBOOT_TRAINING.txt
say "[RUN1] launching our VAE-GAN -> $OUT1"
python training/train_vae_gan.py --resume_vae "$V3" --data_path "$DATA" --gt_subdir ground_truth \
  --critic token --adaptive_adv --gp_mode r1 --conditional_critic \
  --lambda_adv 1.0 --lambda_fm 0 --n_critic 3 \
  --num_epochs 20 --disc_warmup_epochs 2 --lambda_adv_ramp_epochs 3 \
  --ema_start_epoch 2 --batch_size 4 --seed 43 --log_grad_align --point_max_complete 8000 \
  --output_dir "$OUT1" --log_dir "$LOG1" > "$LOG1/run.out" 2>&1
say "[RUN1] training exited rc=$? . final log: $(grep -E 'Epoch ' "$LOG1/run.out" 2>/dev/null | tail -1)"
wait_gpu
say "[RUN1] auto-eval (CD2 vs V3 baseline, deterministic mu-decode, pmc=8000) ..."
python evaluate_vae_gan.py --ckpt "$OUT1/best_vae_gan.pth" --data_path "$DATA" --gt_subdir ground_truth \
  --num_samples 50 --point_max_complete 8000 --use_mean --no_ema \
  --baseline_ckpt "$V3" --output_dir "$OUT1/eval" > "$LOG1/eval.out" 2>&1 \
  && say "[RUN1] eval done -> $OUT1/eval (see $LOG1/eval.out)" \
  || say "[RUN1] eval FAILED (non-fatal); training ckpt is safe. see $LOG1/eval.out"
wait_gpu

# ===================== RUN 2: latent-diffusion V3 pivot =====================
OUT2=checkpoints/latent_diffusion_v3pivot; LOG2=logs/latent_diffusion_v3pivot; mkdir -p "$LOG2"
echo "TRAINING IN PROGRESS - run2 latent diffusion - $(date)" > ~/DO_NOT_REBOOT_TRAINING.txt
say "[RUN2] launching latent diffusion -> $OUT2"
python seed_then_train.py --vae_ckpt "$V3" --data_path "$DATA" --freeze_encoder \
  --num_latent_tokens 32 --num_cond_tokens 32 --hidden_dim 1024 --num_denoiser_blocks 8 --num_heads 4 \
  --num_timesteps 1000 --schedule cosine --denoising_steps 50 \
  --point_max_complete 8000 --point_max_partial 20000 \
  --batch_size 4 --num_epochs 60 --learning_rate 1e-4 --gradient_clip 1.0 \
  --eval_freq 1 --save_freq 5 --output_dir "$OUT2" --log_dir "$LOG2" > "$LOG2/run.out" 2>&1
say "[RUN2] training exited rc=$? ."
wait_gpu
say "[RUN2] auto-eval (CD2 on seq08 v2) ..."
python evaluate_latent_diffusion.py --ckpt "$OUT2/best_latent_diffusion.pth" --data_path "$DATA" \
  --sequence 08 --num_samples 50 --gt_subdir ground_truth --output_dir "$OUT2/eval" > "$LOG2/eval.out" 2>&1 \
  && say "[RUN2] eval done -> $OUT2/eval (see $LOG2/eval.out)" \
  || say "[RUN2] eval FAILED (non-fatal); training ckpt is safe. see $LOG2/eval.out"

say "[queue] ALL DONE. Summary above. Checkpoints: $OUT1 , $OUT2"
echo "ALL QUEUED RUNS FINISHED $(date)" > ~/DO_NOT_REBOOT_TRAINING.txt
