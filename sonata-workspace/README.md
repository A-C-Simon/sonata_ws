## Sonata-LiDiff + VoxFormerDepthPro

Unified workspace for:

- **Sonata-LiDiff** – diffusion-based 3D semantic scene completion on SemanticKITTI.
- **VoxFormerDepthPro** – RGB → Depth Pro → LiDAR-style point clouds for Sonata.

This repo is set up so you can:

- Train Sonata on **real LiDAR** (SemanticKITTI velodyne).
- Train Sonata on **pseudo-LiDAR from RGB** (Depth Pro).
- Compare both pipelines end-to-end with simple commands.

---

## 1. Environment

Create and activate your `sonata_lidiff` environment (if not already):

```bash
conda activate sonata_lidiff
cd ~/Simon_ws/sonata-workspace
pip install -e .
```

Depth Pro is installed separately (see `VoxFormerDepthPro/README.md`), and its checkpoints are under `~/Simon_ws/ml-depth-pro/checkpoints/`.

---

## 2. Datasets

All KITTI / SemanticKITTI archives are assumed to be downloaded to:

```text
~/Simon_ws/dataset/
  data_odometry_color.zip
  data_odometry_velodyne.zip
  data_odometry_labels.zip
  data_odometry_calib.zip
  data_odometry_poses.zip
  data_odometry_voxels.zip
```

They have been extracted (once) into:

```text
~/Simon_ws/dataset/SemanticKITTI/dataset/
  sequences/XX/
    image_2/      # RGB
    velodyne/     # LiDAR
    labels/       # LiDAR labels
    voxels/       # voxel labels + masks
    calib.txt
    poses.txt
  poses/XX.txt    # original KITTI odometry poses
```

> Default paths in the code point to `~/Simon_ws/dataset/SemanticKITTI/dataset` and `~/Simon_ws/dataset/VoxFormerDepthPro_out`, so you rarely need to type them.

For more dataset detail, see `VoxFormerDepthPro/DATASET_AND_RUN.md`.

---

## 3. One‑command pipelines

From the repo root:

```bash
cd ~/Simon_ws/sonata-workspace
conda activate sonata_lidiff
```

### 3.1 LiDAR → Sonata (baseline, original velodyne)

Runs:

1. Generate ground truth maps from **LiDAR** (`map_from_scans.py` with `--save_only_map_world` → one `map_world.npz` per seq).
2. **Plan B:** Precompute voxelized cache (`precompute_voxelized_dataset.py`) → one `.npz` per frame in `voxelized_cache/` for fast training.
3. Train **Sonata-LiDiff diffusion** with `--voxelized_cache_dir` (reads cache, no on-the-fly voxelization).
4. Train **refinement network** on the same data.

Command:

```bash
python scripts/run_lidar_to_sonata.py
```

To run only the voxelized cache pipeline (map + precompute), then train manually:

```bash
python scripts/run_voxelized_cache_semantickitti.py --data_path /path/to/SemanticKITTI/dataset
# If map_world already exists: add --skip_map
# Then run the printed train_diffusion.py command with --voxelized_cache_dir
```

Outputs:

- Checkpoints: `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`
- Logs: `logs/diffusion_lidar`, `logs/refinement_lidar`
- GT: `ground_truth/XX/map_world.npz`; voxelized cache: `voxelized_cache/XX/{scan_id}.npz`

### 3.2 RGB → Depth Pro → Sonata (VoxFormerDepthPro + Sonata)

Runs:

1. **VoxFormerDepthPro**:
   - Preprocess voxel labels.
   - Depth Pro on RGB (`image_2/`) → depth maps (`.npy`).
   - Depth → LiDAR-style point clouds (`.bin`).
   - Assign voxel labels to Depth Pro point clouds.
2. Build a **separate dataset root** for Depth Pro point clouds:
   - `~/Simon_ws/dataset/sonata_depth_pro/` with `sequences/XX/{velodyne,labels,poses.txt,calib.txt}`.
3. Generate **ground truth maps** from this dataset (`map_from_scans.py`, GPU voxelization).
4. Train **Sonata-LiDiff diffusion** and **refinement** on this Depth‑Pro dataset.

Command:

```bash
python scripts/run_depthpro_to_sonata.py
```

Outputs:

- Depth Pro point clouds & labels:
  - `~/Simon_ws/dataset/VoxFormerDepthPro_out/...`
  - `~/Simon_ws/dataset/sonata_depth_pro/sequences/XX/{velodyne,labels,...}`
- GT maps (Depth Pro dataset):
  - `~/Simon_ws/dataset/sonata_depth_pro/ground_truth/XX/*.npz`
- Checkpoints:
  - `checkpoints/diffusion_depthpro`, `checkpoints/refinement_depthpro`
- Logs:
  - `logs/diffusion_depthpro`, `logs/refinement_depthpro`

---

## 4. Individual components (if you want to run them manually)

### 4.1 VoxFormerDepthPro

Entry scripts (defaults use paths in `VoxFormerDepthPro/paths_config.py`):

- `scripts/1_prepare_labels.py` – voxel label preprocessing.
- `scripts/2_run_depth_pro.py` – run Depth Pro on one sequence (`--seq 00` etc.).
- `scripts/3_depth_to_pointcloud.py` – depth → `.bin` for multiple sequences.
- `scripts/4_assign_labels_from_voxels.py` – voxel labels → per-point labels.

See `VoxFormerDepthPro/README.md` and `VoxFormerDepthPro/DATASET_AND_RUN.md` for details.

### 4.2 Ground truth generation (`map_from_scans.py`)

Generates complete scene maps from sequential scans. For **plan B (voxelized cache)** use `--save_only_map_world` so that one `map_world.npz` per sequence is written (needed by `precompute_voxelized_dataset.py`):

```bash
python data/map_from_scans.py \
  --path   ~/Simon_ws/dataset/SemanticKITTI/dataset/sequences \
  --output ~/Simon_ws/dataset/SemanticKITTI/dataset \
  --voxel_size 0.1 \
  --backend torch \
  --save_only_map_world \
  --sequences 00 01 02 03 04 05 06 07 08 09 10
```

- Uses **GPU** if available when `--backend torch`.
- Filters moving-object labels (252–259) when labels are present.

### 4.2.1 Voxelized cache (plan B, faster training)

After `map_from_scans` with `--save_only_map_world`, build the cache and train with it:

```bash
python data/precompute_voxelized_dataset.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir /path/to/SemanticKITTI/dataset/voxelized_cache \
  --voxel_size 0.05 \
  --sequences 00 01 02 03 04 05 06 07 09 10
```

Then pass `--voxelized_cache_dir /path/to/SemanticKITTI/dataset/voxelized_cache` to `train_diffusion.py`. See `docs/VOXELIZED_CACHE_PIPELINE.md` and `docs/LIDAR_BASELINE_VOXELIZATION.md`.

### 4.3 Training scripts

- Diffusion model:

  ```bash
  python training/train_diffusion.py \
    --data_path ~/Simon_ws/dataset/SemanticKITTI/dataset \
    --output_dir checkpoints/diffusion_lidar \
    --log_dir logs/diffusion_lidar
  ```

- Refinement network:

  ```bash
  python training/train_refinement.py \
    --data_path ~/Simon_ws/dataset/SemanticKITTI/dataset \
    --output_dir checkpoints/refinement_lidar \
    --log_dir logs/refinement_lidar
  ```

Replace `--data_path` with `~/Simon_ws/dataset/sonata_depth_pro` to train on Depth Pro instead of LiDAR.

---

## 5. Comparison idea (LiDAR vs RGB→Depth Pro)

After running both orchestration scripts:

- **LiDAR‑trained models**:
  - `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`
- **Depth‑Pro‑trained models**:
  - `checkpoints/diffusion_depthpro`, `checkpoints/refinement_depthpro`

You can now:

- Use the same evaluation code / inference scripts (e.g. `inference.py`, `evaluation/`) with different checkpoints.
- Compare how Sonata performs when:
  - Input = original LiDAR scans vs.
  - Input = point clouds reconstructed from RGB via Depth Pro.

This is the core experiment this workspace is structured to support.

