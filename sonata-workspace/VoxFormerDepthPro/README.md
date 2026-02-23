# VoxFormerDepthPro

VoxFormer preprocessing + Depth Pro for SemanticKITTI, producing LiDAR-style point clouds for Sonata-LiDiff.

## Pipeline

1. **1_prepare_labels** – VoxFormer label preprocessing (voxels → remapped labels .npy)
2. **2_run_depth_pro** – Monocular depth from RGB (Depth Pro) → .npy depth maps
3. **3_depth_to_pointcloud** – Depth maps + calibration → .bin point clouds
4. **4_assign_labels_from_voxels** – Infer per-point labels from voxels (for map_from_scans moving-object filter)

## Requirements

- Sonata-LiDiff environment (PyTorch, numpy, tqdm, pillow, pyyaml, etc.)
- `depth-pro`: `pip install depth-pro`

No `imageio` (replaced by Pillow), no mmcv/mmdet3d.

## Setup

```bash
cd ~/sonata-workspace
pip install -r VoxFormerDepthPro/requirements.txt
```

## Data Layout

- **KITTI root**: SemanticKITTI with `dataset/sequences/XX/{image_2,voxels}/`
- **Calibration**: `data_odometry_calib/sequences/XX/calib.txt`
- **Output**: preprocessed labels, depth maps, point clouds in chosen output dirs

## Usage

```bash
cd VoxFormerDepthPro

# 1. Label preprocessing
python scripts/1_prepare_labels.py -r /path/to/kitti -p /path/to/preprocess

# 2. Depth Pro (per sequence)
python scripts/2_run_depth_pro.py \
  --image_dir /path/to/dataset/sequences/00/image_2 \
  --depth_dir /path/to/depth/sequences/00

# 3. Depth → point clouds
python scripts/3_depth_to_pointcloud.py \
  --depth_root /path/to/depth \
  --calib_root /path/to/data_odometry_calib \
  --save_root /path/to/lidar_pro

# 4. Assign labels from voxels (optional, for LiDiff-style map generation)
python scripts/4_assign_labels_from_voxels.py \
  --pointcloud_root /path/to/lidar_pro \
  --voxel_root /path/to/semantickitti \
  --output_root /path/to/lidar_pro_labeled
```

## Output

- `labels/XX/*.npy` – preprocessed voxel labels (step 1)
- `depth/sequences/XX/*.npy` – metric depth (meters)
- `lidar_pro/sequences/XX/*.bin` – KITTI-style point clouds for Sonata
- `labels/XX/*.label` – per-point labels (step 4). For map_from_scans, copy/symlink to `sequences/XX/labels/` next to velodyne/ and poses.txt.
