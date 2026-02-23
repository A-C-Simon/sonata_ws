"""
Depth map -> LiDAR-style point cloud using KITTI calibration.
Works with Depth Pro depth (meters in rect camera coord).
"""

import os
import numpy as np
from .kitti_util import Calibration


def project_depth_to_velo(calib, depth_map, max_depth=80.0, min_depth=0.1):
    """
    Project depth map to velodyne coordinates.
    depth_map: (H, W) depth in meters (rect camera).
    """
    rows, cols = depth_map.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    depth_flat = depth_map.reshape(-1)
    valid = (depth_flat >= min_depth) & (depth_flat <= max_depth)
    u = c.reshape(-1)[valid]
    v = r.reshape(-1)[valid]
    d = depth_flat[valid]
    uv_depth = np.stack([u.astype(np.float64), v.astype(np.float64), d], axis=1)
    cloud = calib.project_image_to_velo(uv_depth)
    # KITTI velo: front x, left y, up z; filter behind/above
    valid_pts = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_depth)
    return cloud[valid_pts]


def process_sequence(depth_dir, calib_dir, save_dir, max_depth=80.0):
    """Convert all .npy depth maps in depth_dir to .bin point clouds."""
    os.makedirs(save_dir, exist_ok=True)
    calib_file = os.path.join(calib_dir, "calib.txt")
    calib = Calibration(calib_file)
    depth_files = sorted(
        f for f in os.listdir(depth_dir)
        if f.endswith(".npy") and "std" not in f
    )
    for fn in depth_files:
        predix = fn[:-4]
        depth_path = os.path.join(depth_dir, fn)
        depth_map = np.load(depth_path)
        cloud = project_depth_to_velo(calib, depth_map, max_depth=max_depth)
        cloud = np.concatenate([cloud, np.ones((cloud.shape[0], 1))], axis=1)
        cloud = cloud.astype(np.float32)
        out_path = os.path.join(save_dir, predix + ".bin")
        cloud.tofile(out_path)
    return len(depth_files)
