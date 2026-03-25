# Sequence 02 Experiment Report — Training, Inference, and Evaluation

**Dataset frame of reference:** SemanticKITTI sequence `02`, local scene crop **20 m** from the sensor origin, voxel size **0.1 m**.

---

## 1. Objective

Train a **scene completion diffusion** model on an early segment of sequence `02`, run **single-scan inference** on a held-out frame (`000800`), and compare outputs for different **DDIM denoising step** counts against **ground-truth** local maps using **Chamfer** metrics.

---

## 2. Model Pipeline (High Level)

End-to-end stack (`SceneCompletionDiffusion` in `models/diffusion_module.py`):

1. **Input:** One LiDAR scan (or voxelized partial cloud), optionally cropped to a Euclidean ball (`--scene-radius`).
2. **Voxelization:** Partial observation at voxel size `0.1 m` (voxel centers + dummy color/normal for Sonata).
3. **Shell construction:** **Partial voxels + query points** in a cylinder (radius capped by scene radius); training and inference share the same shell logic so counts and geometry match.
4. **Conditioning:** **Sonata** encoder (frozen by default) → **conditional features** mapped to every shell point (concat fusion at level 0).
5. **Denoiser:** **U-Net–style** `DenoisingNetwork` — multi-resolution encoder–decoder with skip links; each block uses a **point-wise** `SonataTransformerBlock` (LayerNorm → grouped features → linear mix → FFN). *Local k-NN attention was removed from this codebase; the denoiser no longer builds a neighbor graph.*
6. **Diffusion:** `num_timesteps=1000`, **cosine** β schedule; training uses random `t` and `q_sample` on the full shell; inference uses **DDIM** (`η=0`) for `num_steps` reverse jumps, with **anchoring** of known partial rows.

**Loss (training):** weighted sum of **Chamfer** (subsampled), **partial consistency**, and **query–query repulsion** (see `SceneCompletionDiffusion.forward`).

**Checkpointing:** `train_diffusion.py` saves `best_model.pth` when **validation loss** improves; `final_model.pth` is always the last epoch.

---

## 3. Training on Seq02 (Frames 0–600)

### 3.1 Split and data window

- **Sequence:** `02` only (`--sequences 02`).
- **Scan index window:** `[0, 601)` → files `000000.bin` … `000600.bin` (**601** frames).  
  *(If you need exactly 600 frames `000000`–`000599`, use `--sequence_scan_end 600`.)*
- **Temporal validation split:** `--intra_seq_val_fraction 0.1` → roughly **90%** earliest scans for train, **10%** latest for val (~**541** / **60** samples in our run).
- **Local scene:** `--scene-radius 20` (meters), aligned with inference.

### 3.2 Hyperparameters (this run)

| Setting | Value |
|--------|--------|
| `voxel_size` | 0.1 |
| `batch_size` | 4 |
| `num_workers` / `prefetch_factor` | 4 / 4 |
| Mixed precision | `--fp16` |
| Epochs | **25** |
| Optimizer | AdamW, lr `1e-4`, cosine schedule per epoch |
| Output dir | `checkpoints/diffusion_seq02_s0_600_r20` |

### 3.3 Training command (reproducibility)

```bash
cd /workspace/sonata_ws/sonata-workspace && PYTHONUNBUFFERED=1 python -u training/train_diffusion.py \
  --data_path /workspace/dataset/SemanticKITTI/dataset \
  --sequences 02 \
  --sequence_scan_start 0 \
  --sequence_scan_end 601 \
  --intra_seq_val_fraction 0.1 \
  --scene-radius 20 \
  --voxel_size 0.1 \
  --batch_size 4 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --fp16 \
  --num_epochs 25 \
  --eval_freq 1 \
  --save_freq 5 \
  --output_dir checkpoints/diffusion_seq02_s0_600_r20 \
  --log_dir logs/diffusion_seq02_s0_600_r20
```

### 3.4 Training outcome (logged)

- **Best validation checkpoint:** `checkpoints/diffusion_seq02_s0_600_r20/best_model.pth`  
  - Recorded at **epoch 21** (0-based in logs).  
  - **Best val loss ≈ 0.1665** (composite training objective, not raw Chamfer alone).
- **Per-epoch wall time:** on the order of **~2–3 minutes** per epoch (GPU-dependent); full **25 epochs** completed in roughly **~1–2 hours** depending on hardware.
- **Train loss** by late epochs was on the order of **~0.15** (same composite loss).

---

## 4. Inference on Frame `000800`

Held-out frame (not in the `[0,601)` training window):  
`/workspace/dataset/SemanticKITTI/dataset/sequences/02/velodyne/000800.bin`

We generated three completions with the **same** `best_model.pth`, varying only `--denoising_steps`:

| Output PLY | Denoising steps |
|------------|-----------------|
| `outputs/seq02_000800_completed_s0_600_train.ply` | 20 (default) |
| `outputs/seq02_000800_completed_s0_600_train_d50.ply` | 50 |
| `outputs/seq02_000800_completed_s0_600_train_d100.ply` | 100 |

### 4.1 Inference command template

```bash
cd /workspace/sonata_ws/sonata-workspace && python -u inference.py \
  --input /workspace/dataset/SemanticKITTI/dataset/sequences/02/velodyne/000800.bin \
  --checkpoint checkpoints/diffusion_seq02_s0_600_r20/best_model.pth \
  --output outputs/seq02_000800_completed_s0_600_train_dXX.ply \
  --scene-radius 20 \
  --voxel_size 0.1 \
  --denoising_steps XX \
  --amp-inference
```

**Observed timing (one GPU, indicative):** total **~50–55 s** per run; **Sonata + model load ~13–14 s**, **`complete_scene` ~38–40 s** (DDIM step count had only a modest effect on wall time in this setup—encoder and fixed shell size dominate).

---

## 5. Quantitative Comparison vs Ground Truth

**GT file:** `ground_truth/02/000800.npz` (key `points`).  
**Preprocessing:** crop GT with **20 m** radius from origin (same as inference).  
**Metric:** symmetric **Chamfer** via `scipy.spatial.cKDTree` — mean squared distance (m²) and mean L2 distance (m).  
**Setup:** fixed random subset of **12,000** GT points (`seed=42`); **full** prediction cloud (**24,000** points).

### 5.1 Results

| Variant | Chamfer mean **sq** dist (m²) ↓ | Chamfer mean **L2** (m) ↓ |
|---------|----------------------------------|----------------------------|
| d20     | **0.286**                        | 0.369                      |
| d100    | 0.289                            | **0.369**                  |
| d50     | 0.294                            | 0.376                      |

**Interpretation:** Differences are **small (~1–3%)**. On this single frame and metric setup, **d20** is slightly better on squared error; **d100** is marginally better on mean distance; **d50** is slightly worse on both. Subsampling GT and fixed shell density limit how definitive this is—repeat with multiple seeds or full GT for stricter benchmarking.

### 5.2 Code to reproduce metrics

```python
"""Chamfer vs GT for seq02 frame 000800 (crop 20 m)."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from utils.point_cloud import crop_lidar_radius

GT_PATH = "/workspace/dataset/SemanticKITTI/dataset/ground_truth/02/000800.npz"
RADIUS = 20.0
SEED = 42
GT_SUB = 12000
ROOT = "/workspace/sonata_ws/sonata-workspace"

plys = {
    "d20": f"{ROOT}/outputs/seq02_000800_completed_s0_600_train.ply",
    "d50": f"{ROOT}/outputs/seq02_000800_completed_s0_600_train_d50.ply",
    "d100": f"{ROOT}/outputs/seq02_000800_completed_s0_600_train_d100.ply",
}

def chamfer(pred, gt_sub):
    tb, ta = cKDTree(gt_sub), cKDTree(pred)
    d_ab, _ = tb.query(pred, k=1)
    d_ba, _ = ta.query(gt_sub, k=1)
    l2_sq = 0.5 * ((d_ab ** 2).mean() + (d_ba ** 2).mean())
    l2 = 0.5 * (d_ab.mean() + d_ba.mean())
    return l2_sq, l2

rng = np.random.default_rng(SEED)
gt = crop_lidar_radius(np.load(GT_PATH)["points"].astype(np.float32), RADIUS)
idx = rng.choice(len(gt), min(GT_SUB, len(gt)), replace=False)
gts = gt[idx].astype(np.float64)

for name, path in plys.items():
    pred = np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)
    c2, c1 = chamfer(pred, gts)
    print(f"{name}: Chamfer m^2={c2:.6f}, Chamfer m={c1:.6f}, N_pred={len(pred)}")
```

Run from `sonata-workspace` with `PYTHONPATH` set so `utils` resolves, e.g.:

```bash
cd /workspace/sonata_ws/sonata-workspace && python -c "<paste snippet adapted as a .py file>"
```

---

## 6. Files and Artifacts (Summary)

| Artifact | Path |
|----------|------|
| Best weights | `checkpoints/diffusion_seq02_s0_600_r20/best_model.pth` |
| Final epoch weights | `checkpoints/diffusion_seq02_s0_600_r20/final_model.pth` |
| Periodic checkpoints | `checkpoints/diffusion_seq02_s0_600_r20/checkpoint_epoch_*.pth` |
| Outputs (frame 800) | `outputs/seq02_000800_completed_s0_600_train*.ply` |

---

## 7. Notes

- **Inference CLI** may force `anchor_alpha=1.0` for partial rows to reduce drift (see console warning during runs).
- **Shell size** at inference matched training-style caps (`num-query-extra` / `max-partial-points` defaults tied to `DEFAULT_TRAIN_MAX_POINTS` in code)—see `inference.py` help for overrides.
- For publication-grade numbers, report **multiple frames**, **confidence intervals**, and optionally **voxel-aligned** or **semantic** metrics in addition to Chamfer.

---

*Report generated to document the seq02 local training window, model pipeline, and single-frame 000800 denoising-step ablation.*
