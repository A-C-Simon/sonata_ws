#!/usr/bin/env python3
"""
VAE-GAN ablation launcher — produces the evidence needed to turn "the GAN
doesn't help" into a defensible result (clean negative OR tradeoff).

It launches a small set of fine-tunes, all warm-started from the SAME converged
VAE V3 checkpoint, with an identical short schedule, varying only the loss:

  recon_only   lambda_adv=0    lambda_fm=0     CONTROL. Pure continued VAE
                                               fine-tuning. This isolates how
                                               much of the reported "6.6% CD² /
                                               4% Hausdorff gain over V3" is just
                                               *more VAE training* vs the GAN.
                                               Every adversarial config must beat
                                               THIS, not raw V3, to claim a win.
  fm_only      lambda_adv=0    lambda_fm=10    Feature-matching only. lambda_fm=10
                                               is ~2 orders larger than the adv
                                               term, so it may drive most of the
                                               regression — this isolates it.
  adv_only_1e-2 lambda_adv=1e-2 lambda_fm=0    Adversarial (Wasserstein) only, at
  adv_only_1e-3 lambda_adv=1e-3 lambda_fm=0    three weights. Sweeps the regime
  adv_only_1e-4 lambda_adv=1e-4 lambda_fm=0    from "strong enough to hurt CD" to
                                               "too weak to matter" — if NONE
                                               helps, the misalignment is not a
                                               tuning artefact.
  full         lambda_adv=1e-2 lambda_fm=10    The shipped config (reference for
                                               the judge critic).

All configs use disc_warmup_epochs=2 + a 3-epoch adv ramp, because we warm-start
from an already-converged V3 — no need for a long recon-only warmup.

This script DOES NOT need a GPU itself; it prints the exact commands by default
(dry run). Pass --execute to run them sequentially on this machine.

After the runs finish, diff them all against the recon_only control:

    python diff_vae_gan_checkpoints.py \
        --ckpt recon_only:checkpoints/vae_gan_abl/recon_only/best_vae_gan.pth \
        --ckpt fm_only:checkpoints/vae_gan_abl/fm_only/best_vae_gan.pth \
        --ckpt adv_1e-2:checkpoints/vae_gan_abl/adv_only_1e-2/best_vae_gan.pth \
        --ckpt adv_1e-3:checkpoints/vae_gan_abl/adv_only_1e-3/best_vae_gan.pth \
        --ckpt adv_1e-4:checkpoints/vae_gan_abl/adv_only_1e-4/best_vae_gan.pth \
        --ckpt full:checkpoints/vae_gan_abl/full/best_vae_gan.pth \
        --judge_ckpt checkpoints/vae_gan_abl/full/best_vae_gan.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "training", "train_vae_gan.py")

# (name, lambda_adv, lambda_fm)
CONFIGS = [
    ("recon_only",    0.0,   0.0),
    ("fm_only",       0.0,   10.0),
    ("adv_only_1e-2", 1e-2,  0.0),
    ("adv_only_1e-3", 1e-3,  0.0),
    ("adv_only_1e-4", 1e-4,  0.0),
    ("full",          1e-2,  10.0),
]


def parse_args():
    p = argparse.ArgumentParser(description="Launch VAE-GAN loss ablations")
    p.add_argument("--resume_vae", type=str,
                   default="checkpoints/point_vae_v3/best_point_vae.pth",
                   help="Converged VAE V3 checkpoint to warm-start every run from.")
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--out_root", type=str, default="checkpoints/vae_gan_abl")
    p.add_argument("--log_root", type=str, default="logs/vae_gan_abl")
    p.add_argument("--num_epochs", type=int, default=25)
    p.add_argument("--disc_warmup_epochs", type=int, default=2)
    p.add_argument("--lambda_adv_ramp_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated subset of config names to run (default: all).")
    p.add_argument("--execute", action="store_true", default=False,
                   help="Actually run the commands sequentially. Default prints them only.")
    return p.parse_args()


def build_cmd(name, lambda_adv, lambda_fm, args):
    out_dir = os.path.join(args.out_root, name)
    log_dir = os.path.join(args.log_root, name)
    cmd = [
        sys.executable, TRAIN,
        "--resume_vae", args.resume_vae,
        "--data_path", args.data_path,
        "--output_dir", out_dir,
        "--log_dir", log_dir,
        "--num_epochs", str(args.num_epochs),
        "--disc_warmup_epochs", str(args.disc_warmup_epochs),
        "--lambda_adv_ramp_epochs", str(args.lambda_adv_ramp_epochs),
        "--lambda_adv", repr(lambda_adv),
        "--lambda_fm", repr(lambda_fm),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--gt_subdir", args.gt_subdir,
    ]
    if args.device:
        cmd += ["--device", args.device]
    return cmd, out_dir


def main():
    args = parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    configs = [c for c in CONFIGS if wanted is None or c[0] in wanted]

    print("=" * 88)
    print("  VAE-GAN ablation plan")
    print(f"  warm-start : {args.resume_vae}")
    print(f"  schedule   : {args.num_epochs} ep, warmup={args.disc_warmup_epochs}, "
          f"ramp={args.lambda_adv_ramp_epochs}, seed={args.seed}")
    print(f"  mode       : {'EXECUTE' if args.execute else 'DRY RUN (print only)'}")
    print("=" * 88)

    produced = []
    for name, l_adv, l_fm in configs:
        cmd, out_dir = build_cmd(name, l_adv, l_fm, args)
        produced.append((name, os.path.join(out_dir, "best_vae_gan.pth")))
        print(f"\n[{name}]  lambda_adv={l_adv}  lambda_fm={l_fm}")
        print("  " + " ".join(cmd))
        if args.execute:
            print(f"  → running {name} ...")
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                print(f"  !! {name} exited with code {r.returncode}; continuing.")

    # Emit the follow-up diff command for convenience.
    full_ckpt = os.path.join(args.out_root, "full", "best_vae_gan.pth")
    diff_cmd = [sys.executable, os.path.join(HERE, "diff_vae_gan_checkpoints.py")]
    for name, ckpt in produced:
        diff_cmd += ["--ckpt", f"{name}:{ckpt}"]
    diff_cmd += ["--judge_ckpt", full_ckpt]

    print("\n" + "=" * 88)
    print("  When runs finish, diff them all against the recon_only control:")
    print("=" * 88)
    print("  " + " ".join(diff_cmd))


if __name__ == "__main__":
    main()
