# VoxFormerDepthPro & Sonata-LiDiff: Dataset Layout & Run Guide

## 1. Downloads Required

| Source | File | Purpose |
|--------|------|---------|
| [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) | `data_odometry_color.zip` (65GB) | RGB images |
| SemanticKITTI | `data_odometry_velodyne.zip` (80GB) | LiDAR scans (for Sonata training) |
| SemanticKITTI | `data_odometry_labels.zip` (179MB) | Per-point LiDAR labels |
| SemanticKITTI | `data_odometry_calib.zip` (1MB) | Calibration (P2, Tr) |
| SemanticKITTI | `data_odometry_poses.zip` (4MB) | Ground truth poses |
| SemanticKITTI | `data_odometry_voxels.zip` | Voxel labels (for VoxFormerDepthPro step 1 & 4) |

Extract all zips into the same folder; they merge into `dataset/`.

**Default paths (no need to type them):** Scripts use `~/Simon_ws/dataset/SemanticKITTI` and `~/Simon_ws/dataset/VoxFormerDepthPro_out` by default. Override with `-r`, `-p`, `--path`, `--data_path`, etc. when needed.

---

## 2. VoxFormerDepthPro: Dataset Layout

Merge extracts so you have **one** SemanticKITTI root (e.g. `$DATA/SemanticKITTI`):

```
$DATA/SemanticKITTI/
├── dataset/
│   └── sequences/
│       ├── 00/
│       │   ├── image_2/       # RGB (from data_odometry_color)
│       │   │   ├── 000000.png
│       │   │   └── ...
│       │   ├── voxels/        # (from data_odometry_voxels)
│       │   │   ├── 000000.label
│       │   │   ├── 000000.invalid
│       │   │   └── ...
│       │   ├── velodyne/      # (optional for VoxFormerDepthPro; needed for Sonata)
│       │   ├── labels/        # (optional for VoxFormerDepthPro; needed for Sonata)
│       │   └── calib.txt      # (from data_odometry_calib)
│       ├── 01/
│       └── ... (00–21)
└── dataset/
    poses/                     # (from data_odometry_poses)
    ├── 00.txt
    ├── 01.txt
    └── ...
```

**Poses per sequence:** `map_from_scans` expects `sequences/XX/poses.txt`. Copy or symlink:

```bash
cd $DATA/SemanticKITTI/dataset
for i in {00..10}; do
  cp poses/${i}.txt sequences/${i}/poses.txt
done
```

---

## 3. VoxFormerDepthPro: Run Commands

Set `KITTI=$DATA/SemanticKITTI` and an output root (e.g. `$OUT/VoxFormerDepthPro`).

```bash
cd ~/Simon_ws/sonata-workspace/VoxFormerDepthPro
conda activate sonata_lidiff

# 1. Label preprocessing (VoxFormer voxels → .npy)
python scripts/1_prepare_labels.py -r $KITTI -p $OUT/VoxFormerDepthPro/preprocess

# 2. Depth Pro (per sequence)
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  python scripts/2_run_depth_pro.py \
    --image_dir $KITTI/dataset/sequences/$seq/image_2 \
    --depth_dir $OUT/VoxFormerDepthPro/depth/sequences/$seq
done

# 3. Depth → point clouds
python scripts/3_depth_to_pointcloud.py \
  --depth_root $OUT/VoxFormerDepthPro/depth \
  --calib_root $KITTI/dataset \
  --save_root $OUT/VoxFormerDepthPro/lidar_pro \
  --sequences 00 01 02 03 04 05 06 07 08 09 10

# 4. Assign labels from voxels (optional, for map_from_scans)
python scripts/4_assign_labels_from_voxels.py \
  -p $OUT/VoxFormerDepthPro/lidar_pro \
  -v $KITTI \
  -o $OUT/VoxFormerDepthPro/lidar_pro_labeled
```

**Note:** Script 3 expects `calib_root` such that `calib_root/sequences/XX/calib.txt` exists. If your calib lives at `$KITTI/dataset/sequences/XX/calib.txt`, use `--calib_root $KITTI/dataset` (parent of `sequences`).

---

## 4. Sonata-LiDiff: Dataset Layout

Sonata expects a **dataset root** whose structure is:

```
$DATA/SemanticKITTI/dataset/   # or $DATA/sonata_dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/          # *.bin (LiDAR or Depth Pro .bin)
│   │   │   ├── 000000.bin
│   │   │   └── ...
│   │   ├── labels/            # *.label (per-point semantics)
│   │   │   ├── 000000.label
│   │   │   └── ...
│   │   ├── poses.txt          # odometry poses (4×4 per line)
│   │   └── calib.txt          # calibration
│   ├── 01/
│   └── ...
└── ground_truth/              # from map_from_scans
    ├── 00/
    │   ├── 000000.npz
    │   ├── 000001.npz
    │   └── ...
    ├── 01/
    └── ...
```

**What Sonata needs:**
- `sequences/XX/velodyne/*.bin` – point clouds (LiDAR or Depth Pro)
- `sequences/XX/labels/*.label` – per-point labels (SemanticKITTI format)
- `sequences/XX/poses.txt` – poses
- `sequences/XX/calib.txt` – calibration
- `ground_truth/XX/*.npz` – GT complete maps (from `map_from_scans`)

---

## 5. Sonata-LiDiff: Run Commands

### A. Use LiDAR (SemanticKITTI velodyne)

1. Arrange layout as above.
2. Copy poses: `cp dataset/poses/XX.txt dataset/sequences/XX/poses.txt`
3. Generate ground truth:
   ```bash
   cd ~/Simon_ws/sonata-workspace
   python data/map_from_scans.py \
     -p $DATA/SemanticKITTI/dataset/sequences \
     -o $DATA/SemanticKITTI/dataset \
     -v 0.1 \
     -b open3d
   ```
4. Train / run inference with `root=$DATA/SemanticKITTI/dataset`.

### B. Use Depth Pro (VoxFormerDepthPro output)

1. Build dataset directory for Sonata:
   ```bash
   SONATA_DATA=$OUT/sonata_depth_pro
   mkdir -p $SONATA_DATA/sequences
   for seq in 00 01 02 03 04 05 06 07 08 09 10; do
     mkdir -p $SONATA_DATA/sequences/$seq/velodyne
     mkdir -p $SONATA_DATA/sequences/$seq/labels
     # Point clouds from VoxFormerDepthPro
     cp $OUT/VoxFormerDepthPro/lidar_pro/sequences/$seq/*.bin $SONATA_DATA/sequences/$seq/velodyne/
     # Labels from step 4
     cp $OUT/VoxFormerDepthPro/lidar_pro_labeled/labels/$seq/*.label $SONATA_DATA/sequences/$seq/labels/
     # Poses & calib from SemanticKITTI
     cp $KITTI/dataset/sequences/$seq/poses.txt $SONATA_DATA/sequences/$seq/
     cp $KITTI/dataset/sequences/$seq/calib.txt $SONATA_DATA/sequences/$seq/
   done
   ```
2. Generate ground truth:
   ```bash
   python data/map_from_scans.py \
     -p $SONATA_DATA/sequences \
     -o $SONATA_DATA \
     -v 0.1 \
     -b open3d
   ```
3. Use `root=$SONATA_DATA` for Sonata training and inference.

---

## 6. Quick Reference: Script Inputs

| Script | Key Paths |
|--------|-----------|
| 1_prepare_labels | `-r` KITTI root (has `dataset/sequences/XX/voxels/`), `-p` preprocess output |
| 2_run_depth_pro | `--image_dir` image folder, `--depth_dir` output |
| 3_depth_to_pointcloud | `--depth_root`, `--calib_root` (parent of `sequences`), `--save_root` |
| 4_assign_labels | `-p` pointcloud root (`sequences/XX/*.bin`), `-v` KITTI root, `-o` output |
| map_from_scans | `-p` sequences folder, `-o` dataset root (writes `ground_truth/`) |
