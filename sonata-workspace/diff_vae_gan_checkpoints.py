#!/usr/bin/env python3
"""
Paired, frame-matched diff of two-or-more VAE-GAN checkpoints on the full RA-L
metric suite — built to settle one question:

    Does the adversarial phase of VAE-GAN actually buy anything, or does it only
    trade away Chamfer fidelity (the metric == the training loss) without
    improving distributional realism / coverage?

Why this driver and not just running evaluate_vae_gan.py twice
--------------------------------------------------------------
1. PAIRED inputs. Running the standalone evaluator twice subsamples the GT with
   an unseeded RNG, so the two runs do NOT see identical inputs and a paired
   significance test is invalid. Here every checkpoint is scored on the *same*
   frame with the *same* subsample (np.random reseeded per-frame) and, by
   default, deterministic mu-decoding (--use_mean). Inputs are bit-identical
   across checkpoints → a clean paired Wilcoxon signed-rank test.

2. A SINGLE FIXED JUDGE CRITIC. A checkpoint's own critic is meaningless for the
   pre-adversarial model (its disc is still untrained), so per-checkpoint critic
   scores are not comparable. Instead we load ONE trained critic (the
   post-adversarial one by default, --judge_ckpt) and use it to score every
   model's reconstructions. This answers "does the adversarial model actually
   produce clouds the trained critic finds more realistic than the pre-adv
   model?" — an apples-to-apples realism test.

3. AUTOMATED VERDICT. Fidelity (CD²) is directly optimised by the VAE, so the
   GAN cannot beat the pre-adv model on it by construction; the scientific
   content is in the *other* axes. The driver classifies the result:
       - GAN HELPS          : CD² improves significantly.
       - CLEAN NEGATIVE      : CD² regresses, and NOTHING (JSD, coverage, IoU,
                               Hausdorff, judged realism) improves significantly.
       - TRADEOFF            : CD² regresses but ≥1 distributional/realism metric
                               improves significantly → reframe, don't call it a
                               failure.
       - INCONCLUSIVE        : no significant fidelity change.

Usage
-----
    python diff_vae_gan_checkpoints.py \
        --ckpt pre_adv:checkpoints/vae_gan/best_vae_gan.pth \
        --ckpt post_adv:checkpoints/vae_gan/checkpoint_epoch_29.pth \
        --num_samples 50 --sequence 08 \
        --output_dir diff_vae_gan

    # N-way (e.g. the ablation sweep), first --ckpt is the reference:
    python diff_vae_gan_checkpoints.py \
        --ckpt recon_only:checkpoints/vae_gan_abl/recon_only/best_vae_gan.pth \
        --ckpt adv_1e-3:checkpoints/vae_gan_abl/adv_only_1e-3/best_vae_gan.pth \
        --ckpt fm_only:checkpoints/vae_gan_abl/fm_only/best_vae_gan.pth \
        --judge_ckpt checkpoints/vae_gan_abl/full/best_vae_gan.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.discriminator import MultiScalePointDiscriminator
from models.point_cloud_vae import PointCloudVAE
# Reuse the exact single-frame pipeline + IO helpers from the standalone eval so
# normalisation/decoding stays byte-for-byte identical to evaluate_vae_gan.py.
from evaluate_vae_gan import evaluate_frame, load_bin, load_gt


# ---------------------------------------------------------------------------
# Metric registry: which way is "better", and how to format
# ---------------------------------------------------------------------------
# (key, pretty label, lower_is_better, scale, fmt)
METRICS = [
    ("cd_norm",      "CD² normalised (train/val metric)", True,  1.0, ".6f"),
    ("cd_sq",        "CD² original (m²)",                 True,  1.0, ".6f"),
    ("cd",           "CD linear (m)",                     True,  1.0, ".4f"),
    ("jsd",          "JSD-BEV 0.5m",                      True,  1.0, ".4f"),
    ("hausdorff_95", "Hausdorff-95 (m)",                  True,  1.0, ".4f"),
    ("f_score@0.1",  "F-score @0.1m",                     False, 1.0, ".4f"),
    ("f_score@0.2",  "F-score @0.2m",                     False, 1.0, ".4f"),
    ("precision@0.1","Precision @0.1m",                   False, 1.0, ".4f"),
    ("recall@0.1",   "Recall @0.1m",                      False, 1.0, ".4f"),
    ("iou@0.1",      "Voxel IoU @0.1m",                   False, 1.0, ".4f"),
    ("iou@0.2",      "Voxel IoU @0.2m",                   False, 1.0, ".4f"),
    # Realism, judged by the single fixed critic. fake_score higher == more
    # realistic to the judge; gap (real-fake) lower == better fooling.
    ("fake_score",   "Judge critic D(fake) ↑realism",     False, 1.0, ".4f"),
    ("critic_gap",   "Judge realism gap (real−fake)",     True,  1.0, ".4f"),
]

# Metrics that count as "distributional realism / coverage" for the verdict
# (i.e. axes a GAN could legitimately improve even while CD² regresses).
REALISM_KEYS = [
    "jsd", "hausdorff_95",
    "f_score@0.1", "f_score@0.2", "precision@0.1", "recall@0.1",
    "iou@0.1", "iou@0.2",
    "fake_score", "critic_gap",
]
FIDELITY_KEY = "cd_sq"  # headline fidelity metric for the verdict


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def build_vae(args, device):
    return PointCloudVAE(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        num_latent_tokens=args.num_latent_tokens,
        internal_dim=args.internal_dim,
        num_heads=args.num_heads,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)


def load_vae_from_ckpt(path, args, device, prefer_ema=True):
    """Load a VAE, preferring EMA shadow weights when present (publication-grade).

    Also supports Stage-2 residual checkpoints (refiner_state_dict): builds the
    frozen base VAE from ckpt['resume_vae'] and wraps it with the ResidualRefiner
    as a RefinedVAE that presents the same interface.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "refiner_state_dict" in ckpt:
        from models.residual_refine import ResidualRefiner, RefinedVAE
        base, _, _ = load_vae_from_ckpt(ckpt["resume_vae"], args, device, prefer_ema=False)
        for p in base.parameters():
            p.requires_grad_(False)
        refiner = ResidualRefiner(dim=args.internal_dim, num_heads=args.num_heads,
                                  num_blocks=ckpt.get("refiner_blocks", 2),
                                  max_offset=ckpt.get("max_offset", 0.1)).to(device)
        refiner.load_state_dict(ckpt["refiner_state_dict"])
        refiner.eval()
        wrapped = RefinedVAE(base, refiner).to(device).eval()
        return wrapped, "residual", ckpt.get("epoch", "?")
    vae = build_vae(args, device)
    if prefer_ema and "vae_ema_state_dict" in ckpt:
        vae.load_state_dict(ckpt["vae_ema_state_dict"])
        src = "EMA"
    elif "vae_state_dict" in ckpt:
        vae.load_state_dict(ckpt["vae_state_dict"])
        src = "live"
    else:
        # bare V3-style checkpoint (model_state_dict or raw state dict)
        vae.load_state_dict(ckpt.get("model_state_dict", ckpt))
        src = "raw"
    vae.eval()
    return vae, src, ckpt.get("epoch", "?")


def load_judge_disc(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "disc_state_dict" not in ckpt:
        return None
    disc = MultiScalePointDiscriminator(in_dim=3).to(device)
    disc.load_state_dict(ckpt["disc_state_dict"])
    disc.eval()
    return disc


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def _paired_wilcoxon(ref, cand):
    """
    Two-sided Wilcoxon signed-rank on paired arrays. Returns (statistic, pvalue).
    Degrades gracefully: identical arrays / all-zero diffs / missing scipy →
    (nan, 1.0). Direction is reported separately from the mean delta, so a
    two-sided p-value is all we need here.
    """
    ref = np.asarray(ref, float)
    cand = np.asarray(cand, float)
    diff = cand - ref
    if diff.size == 0 or np.allclose(diff, 0.0):
        return float("nan"), 1.0
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(ref, cand)  # two-sided; zeros handled by default
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _agg(values):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def significance_stars(p):
    if np.isnan(p):
        return "  "
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "** "
    if p < 5e-2:
        return "*  "
    return "ns "


# ---------------------------------------------------------------------------
# Evaluation loop (all checkpoints, frame-paired)
# ---------------------------------------------------------------------------

def collect(models, judge, frames, vel_dir, gt_dir, eval_args, device, seed):
    """
    Returns per_frame[label] = list of metric dicts, frame-aligned across labels.
    Every model sees an identical subsample for a given frame (np.random reseed),
    and is judged by the same fixed critic.
    """
    per_frame = {label: [] for label, _, _, _ in models}
    used_frames = []

    for fi, fid in enumerate(frames):
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        if not os.path.exists(gt_path) or not os.path.exists(lidar_path):
            continue
        lidar_raw = load_bin(lidar_path)
        gt_raw = load_gt(gt_path)

        ok = True
        frame_results = {}
        for label, vae, _, _ in models:
            # Identical input per frame across all checkpoints → valid pairing.
            np.random.seed(seed + fi)
            torch.manual_seed(seed + fi)
            try:
                res, _ = evaluate_frame(vae, judge, lidar_raw, gt_raw, eval_args, device)
            except Exception as e:  # noqa: BLE001
                print(f"  ! frame {fid} / {label}: {e}")
                ok = False
                break
            # Derive realism gap from the single judge critic.
            if res.get("real_score") is not None and res.get("fake_score") is not None:
                res["critic_gap"] = res["real_score"] - res["fake_score"]
            else:
                res["critic_gap"] = float("nan")
            frame_results[label] = res
        if not ok:
            continue

        for label in frame_results:
            per_frame[label].append(frame_results[label])
        used_frames.append(fid)

        gaps = "  ".join(
            f"{label}:CD²={frame_results[label]['cd_sq']:.4f}"
            for label, *_ in models
        )
        print(f"  [{len(used_frames):3d}] {fid}  {gaps}")

    return per_frame, used_frames


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(models, per_frame, used_frames, ref_label, args):
    labels = [m[0] for m in models]

    # aggregate per label
    aggregated = {label: {} for label in labels}
    arrays = {label: {} for label in labels}
    for label in labels:
        rows = per_frame[label]
        for key, *_ in METRICS:
            vals = [r.get(key) for r in rows]
            arrays[label][key] = vals
            mean, std = _agg(vals)
            aggregated[label][key] = {"mean": mean, "std": std}

    # pairwise stats vs reference
    pairwise = {}
    for label in labels:
        if label == ref_label:
            continue
        pairwise[label] = {}
        for key, _, lower_better, _, _ in METRICS:
            ref_vals = arrays[ref_label][key]
            cand_vals = arrays[label][key]
            paired = [
                (a, b) for a, b in zip(ref_vals, cand_vals)
                if a is not None and b is not None
                and not np.isnan(a) and not np.isnan(b)
            ]
            if not paired:
                pairwise[label][key] = dict(delta=float("nan"), pct=float("nan"),
                                            p=float("nan"), n=0,
                                            improved=False, significant=False)
                continue
            ra = np.array([p[0] for p in paired])
            ca = np.array([p[1] for p in paired])
            delta = float(ca.mean() - ra.mean())
            base = ra.mean()
            pct = float(100.0 * delta / base) if abs(base) > 1e-12 else float("nan")
            _, p = _paired_wilcoxon(ra, ca)
            improved = (delta < 0) if lower_better else (delta > 0)
            significant = (not np.isnan(p)) and (p < 0.05)
            pairwise[label][key] = dict(
                delta=delta, pct=pct, p=p, n=len(paired),
                improved=bool(improved), significant=bool(significant),
            )

    # verdict (only meaningful for the first non-reference candidate; reported
    # per candidate)
    verdicts = {}
    for label in labels:
        if label == ref_label:
            continue
        st = pairwise[label]
        fid_stat = st.get(FIDELITY_KEY, {})
        fid_regressed = fid_stat.get("significant") and not fid_stat.get("improved")
        fid_improved = fid_stat.get("significant") and fid_stat.get("improved")
        realism_wins = [
            k for k in REALISM_KEYS
            if st.get(k, {}).get("significant") and st.get(k, {}).get("improved")
        ]
        if fid_improved:
            verdict = ("GAN_HELPS",
                       f"{label}: CD² improves significantly vs {ref_label} — positive result.")
        elif fid_regressed and not realism_wins:
            verdict = ("CLEAN_NEGATIVE",
                       f"{label}: CD² regresses significantly and NO distributional/realism "
                       f"metric improves → clean negative result. The adversarial objective "
                       f"trades fidelity for nothing measurable.")
        elif fid_regressed and realism_wins:
            verdict = ("TRADEOFF",
                       f"{label}: CD² regresses but these improve significantly: "
                       f"{', '.join(realism_wins)} → fidelity-vs-realism tradeoff, not a failure.")
        else:
            verdict = ("INCONCLUSIVE",
                       f"{label}: no significant CD² change vs {ref_label}.")
        verdicts[label] = {"code": verdict[0], "text": verdict[1],
                           "realism_wins": realism_wins}

    return {
        "config": {
            "checkpoints": {m[0]: m[3] for m in models},
            "weight_source": {m[0]: m[2] for m in models},
            "reference": ref_label,
            "judge_ckpt": args.judge_ckpt,
            "num_frames": len(used_frames),
            "sequence": args.sequence,
            "use_mean": args.use_mean,
            "seed": args.seed,
            "frames": used_frames,
        },
        "aggregated": aggregated,
        "pairwise_vs_reference": pairwise,
        "verdicts": verdicts,
    }


def print_report(report, models, ref_label):
    labels = [m[0] for m in models]
    agg = report["aggregated"]
    pw = report["pairwise_vs_reference"]
    cfg = report["config"]

    print("\n" + "=" * 100)
    print(f"  VAE-GAN checkpoint diff — {cfg['num_frames']} paired frames, seq {cfg['sequence']}")
    print(f"  Reference: {ref_label}   |   Judge critic: {cfg['judge_ckpt']}")
    print(f"  Decoding : {'deterministic μ (use_mean)' if cfg['use_mean'] else 'stochastic z'}"
          f"   seed={cfg['seed']}")
    for label in labels:
        ws = cfg["weight_source"][label]
        print(f"    {label:<16s} {cfg['checkpoints'][label]}  [{ws} weights]")
    print("=" * 100)

    # header
    col_w = 20
    head = f"  {'metric':<34s}" + "".join(f"{lab[:col_w-1]:>{col_w}}" for lab in labels)
    print(head)
    print("  " + "-" * (34 + col_w * len(labels) - 2))

    for key, label, lower_better, scale, fmt in METRICS:
        arrow = "↓" if lower_better else "↑"
        row = f"  {label + ' ' + arrow:<34s}"
        for lab in labels:
            m = agg[lab][key]["mean"] * scale
            cell = "nan" if np.isnan(m) else format(m, fmt)
            row += f"{cell:>{col_w}}"
        print(row)
        # delta line vs reference
        dline = f"  {'   Δ vs ' + ref_label:<34s}"
        for lab in labels:
            if lab == ref_label:
                dline += f"{'—':>{col_w}}"
                continue
            st = pw[lab][key]
            d = st["delta"]
            if np.isnan(d):
                dline += f"{'nan':>{col_w}}"
                continue
            mark = "+" if st["improved"] else "-"   # +=better, -=worse (direction-aware)
            stars = significance_stars(st["p"]).strip()
            cell = f"{mark}{format(abs(d), fmt)}{stars}"
            dline += f"{cell:>{col_w}}"
        print(dline)

    print("\n  Legend: ↓ lower better, ↑ higher better. Δ sign is direction-aware "
          "(+ = better than ref, - = worse).")
    print("  Significance (two-sided paired Wilcoxon): *** p<1e-3  ** p<1e-2  * p<5e-2  (blank = ns)")

    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    for lab, v in report["verdicts"].items():
        print(f"  [{v['code']}] {v['text']}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_ckpt_spec(spec, idx):
    """'label:path' or bare 'path' (auto-labelled)."""
    if ":" in spec and not os.path.exists(spec):
        label, path = spec.split(":", 1)
        return label, path
    # allow 'label:path' even when path contains no ':'
    if ":" in spec:
        label, path = spec.split(":", 1)
        if os.path.exists(path):
            return label, path
    return f"ckpt{idx}", spec


def parse_args():
    p = argparse.ArgumentParser(description="Paired full-metric diff of VAE-GAN checkpoints")
    p.add_argument("--ckpt", action="append", required=True,
                   help="Repeatable. 'label:path'. The FIRST is the reference baseline.")
    p.add_argument("--judge_ckpt", type=str, default=None,
                   help="Checkpoint whose discriminator scores ALL models (fixed realism judge). "
                        "Default: the LAST --ckpt provided (the post-adversarial one).")
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--sequence", type=str, default="08")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--output_dir", type=str, default="diff_vae_gan")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for per-frame subsampling (ensures identical inputs across ckpts).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_ema", action="store_true", default=False,
                   help="Use live weights instead of EMA shadow weights.")
    p.add_argument("--stochastic", dest="use_mean", action="store_false", default=True,
                   help="Sample z instead of using μ. Default is deterministic μ (use_mean) "
                        "so the comparison is exactly paired.")
    # VAE architecture (must match training)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    specs = [parse_ckpt_spec(s, i) for i, s in enumerate(args.ckpt)]
    if len(specs) < 2:
        raise SystemExit("Need at least two --ckpt specs to diff.")
    ref_label = specs[0][0]
    if args.judge_ckpt is None:
        args.judge_ckpt = specs[-1][1]
        print(f"No --judge_ckpt given; using last checkpoint's critic as judge: {args.judge_ckpt}")

    # Load all VAEs.
    models = []  # (label, vae, weight_source, path)
    for label, path in specs:
        vae, src, epoch = load_vae_from_ckpt(path, args, device, prefer_ema=not args.no_ema)
        print(f"Loaded {label:<16s} epoch={epoch}  [{src} weights]  {path}")
        models.append((label, vae, src, path))

    # Single fixed judge critic.
    judge = load_judge_disc(args.judge_ckpt, device)
    if judge is None:
        print(f"WARNING: judge checkpoint {args.judge_ckpt} has no disc_state_dict — "
              f"realism metrics will be NaN.")

    # eval args namespace consumed by evaluate_frame
    eval_args = SimpleNamespace(
        point_max_complete=args.point_max_complete,
        use_mean=args.use_mean,
        n_stochastic=1,
    )

    # frames
    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, args.gt_subdir, args.sequence)
    all_frames = sorted(f[:-4] for f in os.listdir(vel_dir) if f.endswith(".bin"))
    step = max(1, len(all_frames) // args.num_samples)
    frames = all_frames[::step][: args.num_samples]
    print(f"\nDiffing {len(specs)} checkpoints over up to {len(frames)} frames "
          f"(deterministic={'yes' if args.use_mean else 'no'})\n")

    per_frame, used = collect(models, judge, frames, vel_dir, gt_dir, eval_args, device, args.seed)
    if not used:
        raise SystemExit("No frames evaluated — check --data_path / --gt_subdir / --sequence.")

    report = build_report(models, per_frame, used, ref_label, args)
    report["per_frame"] = {lab: per_frame[lab] for lab in per_frame}
    print_report(report, models, ref_label)

    out = os.path.join(args.output_dir, "diff_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report → {out}")


if __name__ == "__main__":
    main()
