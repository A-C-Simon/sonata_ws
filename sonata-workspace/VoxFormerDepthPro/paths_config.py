"""
Default paths for VoxFormerDepthPro. Use env dataset and OUT to override (e.g. dataset=/workspace/dataset).
KITTI = $dataset/SemanticKITTI, outputs = $OUT/VoxFormerDepthPro.
"""
import os

# Dataset root: env "dataset" or default ~/Simon_ws/dataset
WORKSPACE_DATASET = os.environ.get("dataset") or os.path.expanduser("~/Simon_ws/dataset")
# SemanticKITTI root (KITTI = $dataset/SemanticKITTI)
DEFAULT_KITTI_ROOT = os.path.join(WORKSPACE_DATASET, "SemanticKITTI")
DATASET_SUBDIR = "dataset"

# Output root: env "OUT" or same as dataset root. VoxFormerDepthPro out = $OUT/VoxFormerDepthPro
OUT_ROOT = os.environ.get("OUT") or WORKSPACE_DATASET
DEFAULT_VOXFORMER_OUT = os.path.join(OUT_ROOT, "VoxFormerDepthPro")


def get_dataset_root():
    return os.path.join(DEFAULT_KITTI_ROOT, DATASET_SUBDIR)


def get_sequences_path():
    return os.path.join(get_dataset_root(), "sequences")


def get_preprocess_root():
    return os.path.join(DEFAULT_VOXFORMER_OUT, "preprocess")


def get_depth_root():
    return os.path.join(DEFAULT_VOXFORMER_OUT, "depth")


def get_lidar_pro_root():
    return os.path.join(DEFAULT_VOXFORMER_OUT, "lidar_pro")


def get_lidar_pro_labeled_root():
    return os.path.join(DEFAULT_VOXFORMER_OUT, "lidar_pro_labeled")
