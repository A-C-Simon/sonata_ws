# Report: diffusion teacher evaluation (`evaluate.py`)

**Date:** 2025-03-25

## Purpose

Sanity-check the diffusion **teacher** checkpoint with `evaluate.py`: one-step `x₀` prediction at `t=200`, Chamfer distance vs. GT, BEV renderings.

> This is **not** full inference (`inference.py` / DDIM `complete_scene`).

## Shared configuration

| Parameter | Value |
|-----------|--------|
| Script | `evaluate.py` |
| Checkpoint | `checkpoints/diffusion_teacher_lidar_val0.213_epoch20.pth` (epoch 20) |
| Dataset root | `/workspace/dataset/SemanticKITTI/dataset` |
| Sequence | `08` |
| Frame sampling | Uniform stride over the sorted `velodyne` list (`--num_samples`); below: **3** and **20** frames |
| Model | `SceneCompletionDiffusion`, Sonata `feature_levels=[0]`, **`conditioning_mode=additive`**, `load_state_dict(..., strict=False)` (checkpoint has no `cond_fuse` weights) |
| Preprocessing | Voxelized partial, size `0.05`, center = **mean(LiDAR)**; `gt_target` is GT voxelized with the same center |

Chamfer: `refinement_net_2.chamfer_distance` — symmetric mean over **squared** distances (not raw meters).  
**Mean ± std** across frames matches the script (std over the per-frame CD list).

### Comparing CHD across `ground_truth` vs. `ground_truth_v2`

**`ground_truth_v2` has far fewer points** (sparser map / different density than baseline `ground_truth`). Chamfer depends on how many points the reference has and where they lie: with a **sparser GT**, the prediction does not need to match as many “tight” locations, so **CHD can be systematically lower** for the same model geometry. You should **not** read e.g. **0.4 vs. 0.06 as “the model is better on v2”** without accounting for GT density — these are mainly **different references**. Prefer tracking metrics **within one GT variant**, normalizing density (fixed downsampling / matched target point counts), and/or adding metrics less sensitive to large shifts in reference point count.

---

## Three frames (`--num_samples 3`)

Same three indices as the short smoke test (first, middle, last in the size-3 subsample).

### Run 1 — GT `ground_truth`

| Parameter | Value |
|-----------|--------|
| GT | `ground_truth/08/<frame>.npz` |
| Output | `evaluation_results_teacher/` |

| Frame | Teacher Chamfer | Time (s) |
|-------|-----------------|----------|
| 000000 | 0.8290 | 1.41 |
| 001357 | 0.3661 | 1.52 |
| 002714 | 0.4809 | 1.40 |

**Mean teacher CD:** 0.5587 ± 0.1968  
**Mean time:** ~1.44 s / frame

### Run 2 — GT `ground_truth_v2`

| Parameter | Value |
|-----------|--------|
| GT | `ground_truth_v2/08/<frame>.npz` (`--gt_subdir ground_truth_v2`) |
| Output | `evaluation_results_teacher_gt_v2/` |

| Frame | Teacher Chamfer | Time (s) |
|-------|-----------------|----------|
| 000000 | 0.1125 | 1.59 |
| 001357 | 0.0634 | 1.40 |
| 002714 | 0.0633 | 1.39 |

**Mean teacher CD:** 0.0797 ± 0.0232  
**Mean time:** ~1.46 s / frame

---

## Twenty frames (`--num_samples 20`)

Uniform stride over `velodyne`: frames `000000` … `003857` (index step 203 in the file list).

### Run 1 — GT `ground_truth`

| Parameter | Value |
|-----------|--------|
| Output | `evaluation_results_teacher_n20/` |

| Frame | Teacher Chamfer | Time (s) | Frame | Teacher Chamfer | Time (s) |
|-------|-----------------|----------|-------|-----------------|----------|
| 000000 | 0.8133 | 1.62 | 002030 | 0.2769 | 1.40 |
| 000203 | 0.6537 | 1.41 | 002233 | 0.2453 | 1.40 |
| 000406 | 0.6442 | 1.42 | 002436 | 0.2457 | 1.40 |
| 000609 | 0.2203 | 1.40 | 002639 | 0.3043 | 1.42 |
| 000812 | 0.3645 | 1.41 | 002842 | 0.5158 | 1.46 |
| 001015 | 0.4117 | 1.40 | 003045 | 0.6370 | 1.42 |
| 001218 | 0.1978 | 1.40 | 003248 | 0.4736 | 1.42 |
| 001421 | 0.3062 | 1.45 | 003451 | 0.3634 | 1.39 |
| 001624 | 0.5197 | 1.40 | 003654 | 0.2932 | 1.40 |
| 001827 | 0.5396 | 1.41 | 003857 | 0.3086 | 1.41 |

**Mean teacher CD:** 0.4167 ± 0.1695  
**Mean time:** 1.42 s / frame

### Run 2 — GT `ground_truth_v2`

| Parameter | Value |
|-----------|--------|
| Output | `evaluation_results_teacher_gt_v2_n20/` |

| Frame | Teacher Chamfer | Time (s) | Frame | Teacher Chamfer | Time (s) |
|-------|-----------------|----------|-------|-----------------|----------|
| 000000 | 0.1116 | 1.59 | 002030 | 0.0641 | 1.42 |
| 000203 | 0.0591 | 1.69 | 002233 | 0.0475 | 1.43 |
| 000406 | 0.0619 | 1.55 | 002436 | 0.0504 | 1.43 |
| 000609 | 0.0652 | 1.40 | 002639 | 0.0577 | 1.40 |
| 000812 | 0.0624 | 1.43 | 002842 | 0.0640 | 1.40 |
| 001015 | 0.0562 | 1.41 | 003045 | 0.0567 | 1.40 |
| 001218 | 0.0665 | 1.40 | 003248 | 0.0681 | 1.41 |
| 001421 | 0.0595 | 1.40 | 003451 | 0.0690 | 1.59 |
| 001624 | 0.0554 | 1.42 | 003654 | 0.0591 | 1.40 |
| 001827 | 0.0530 | 1.41 | 003857 | 0.0599 | 1.40 |

**Mean teacher CD:** 0.0624 ± 0.0126  
**Mean time:** 1.45 s / frame

### Summary: N=3 vs N=20

| GT | N=3 mean ± std | N=20 mean ± std |
|----|----------------|-----------------|
| `ground_truth` | 0.5587 ± 0.1968 | 0.4167 ± 0.1695 |
| `ground_truth_v2` | 0.0797 ± 0.0232 | 0.0624 ± 0.0126 |

The table **does not rank v1 vs v2 model quality** — see **Comparing CHD** above (sparser v2 GT often yields lower CHD regardless of whether the prediction is “better”). Within each GT: for v1, **N=20** mean is below the three-frame subset (different indices + more typical average); for v2, spread stays tight around ~0.06.

---

## GT comparison (short)

Beyond point density: check **which GT variant the teacher was trained on** and which map better matches the partial scan semantically (corridors, voxel occupancy, etc.).

---

## Artifacts

| Run | Directory |
|-----|-----------|
| GT v1, N=3 | `evaluation_results_teacher/` |
| GT v2, N=3 | `evaluation_results_teacher_gt_v2/` |
| GT v1, N=20 | `evaluation_results_teacher_n20/` |
| GT v2, N=20 | `evaluation_results_teacher_gt_v2_n20/` |

Each run includes `metrics.json` and `bev_<frame>.png`.

---

## Reproduction

**GT v1, 3 frames:**
```bash
cd /workspace/sonata_ws/sonata-workspace
python3 evaluate.py \
  --data_path /workspace/dataset/SemanticKITTI/dataset \
  --teacher_ckpt checkpoints/diffusion_teacher_lidar_val0.213_epoch20.pth \
  --output_dir evaluation_results_teacher \
  --num_samples 3 --sequence 08
```

**GT v2, 3 frames:**
```bash
python3 evaluate.py \
  --data_path /workspace/dataset/SemanticKITTI/dataset \
  --teacher_ckpt checkpoints/diffusion_teacher_lidar_val0.213_epoch20.pth \
  --output_dir evaluation_results_teacher_gt_v2 \
  --num_samples 3 --sequence 08 \
  --gt_subdir ground_truth_v2
```

**GT v1, 20 frames:**
```bash
python3 evaluate.py \
  --data_path /workspace/dataset/SemanticKITTI/dataset \
  --teacher_ckpt checkpoints/diffusion_teacher_lidar_val0.213_epoch20.pth \
  --output_dir evaluation_results_teacher_n20 \
  --num_samples 20 --sequence 08
```

**GT v2, 20 frames:**
```bash
python3 evaluate.py \
  --data_path /workspace/dataset/SemanticKITTI/dataset \
  --teacher_ckpt checkpoints/diffusion_teacher_lidar_val0.213_epoch20.pth \
  --output_dir evaluation_results_teacher_gt_v2_n20 \
  --num_samples 20 --sequence 08 \
  --gt_subdir ground_truth_v2
```

If filenames are `000000_v2.npz`, add: `--gt_suffix _v2`.
