#!/usr/bin/env bash
# Wait for the Stage-2 residual run to finish, then run the residual-aware
# paired-Wilcoxon diff of the refiner vs control / v3 / Stage-1 fix.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0

RES_PID=1586328
echo "[eval-queue] waiting for residual PID $RES_PID ... $(date)"
while kill -0 "$RES_PID" 2>/dev/null; do sleep 60; done
echo "[eval-queue] residual run finished, starting eval at $(date)"

RES=checkpoints/residual_gan_v2/cdleash_uncond
FIX=checkpoints/vae_gan_fix_v2/token_adapt_r1_cond
CTRL=checkpoints/vae_gan_abl_v2/recon_only/best_vae_gan.pth

if [ ! -f "$RES/best_residual.pth" ]; then
  echo "[eval-queue] ERROR: $RES/best_residual.pth not found; aborting eval."
  exit 1
fi

ARGS=(--no_ema --num_samples 300 --seed 0 --gt_subdir ground_truth
  --ckpt control:$CTRL
  --ckpt v3:checkpoints/point_vae_v3/best_point_vae.pth
  --ckpt fix_ep14:$FIX/checkpoint_epoch_14.pth
  --ckpt res_best:$RES/best_residual.pth
  --judge_ckpt $CTRL
  --output_dir diff_residual_v2_eval)
# include the latest periodic refiner checkpoint (most adv-trained) if present
LAST=$(ls -1 "$RES"/refiner_epoch_*.pth 2>/dev/null | sort -t_ -k3 -n | tail -1)
[ -n "$LAST" ] && ARGS+=(--ckpt res_last:"$LAST")

python diff_vae_gan_checkpoints.py "${ARGS[@]}"
echo "[eval-queue] eval finished at $(date) (exit $?)"
