"""
Default paths for VoxFormerDepthPro so you don't have to type them every run.
Override any path via script arguments.
"""
import os

# Workspace dataset root (SemanticKITTI extracted here)
WORKSPACE_DATASET = os.path.expanduser("~/Simon_ws/dataset")

# SemanticKITTI root: contains dataset/sequences/XX/{image_2,velodyne,labels,voxels}, dataset/poses/
DEFAULT_KITTI_ROOT = os.path.join(WORKSPACE_DATASET, "SemanticKITTI")

# Dataset folder inside KITTI root (has sequences/ and poses/)
DATASET_SUBDIR = "dataset"

# VoxFormerDepthPro outputs (separate from Sonata GT)
DEFAULT_VOXFORMER_OUT = os.path.join(WORKSPACE_DATASET, "VoxFormerDepthPro_out")

# Derived: dataset root for calib, sequences path for map_from_scans
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
