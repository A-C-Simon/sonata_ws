"""
Assign per-point labels to Depth Pro point clouds by looking up voxel labels.

Both voxels and Depth Pro points are in the same sensor/velodyne frame per frame.
SemanticKITTI voxel grid: 256x256x32, 0.2m resolution.
Physical range: X [0, 51.2), Y [-25.6, 25.6), Z [0, 6.4) meters.
"""

import os
import numpy as np

# SemanticKITTI SSC voxel grid
VOXEL_RES = 0.2
GRID_SHAPE = (256, 256, 32)  # X, Y, Z
X_RANGE = (0.0, 51.2)
Y_RANGE = (-25.6, 25.6)
Z_RANGE = (0.0, 6.4)


def world_to_voxel_index(points: np.ndarray) -> np.ndarray:
    """
    Map velodyne points (N, 3) to voxel indices (N, 3).
    Returns int32 indices; out-of-range -> -1 for that axis.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    vx = np.floor(x / VOXEL_RES).astype(np.int32)
    vy = np.floor((y - Y_RANGE[0]) / VOXEL_RES).astype(np.int32)
    vz = np.floor((z - Z_RANGE[0]) / VOXEL_RES).astype(np.int32)
    return np.stack([vx, vy, vz], axis=1)


def assign_labels_from_voxel_grid(
    points: np.ndarray,
    voxel_labels: np.ndarray,
) -> np.ndarray:
    """
    Assign semantic label to each point by voxel lookup.

    Args:
        points: (N, 3) in velodyne frame
        voxel_labels: (256, 256, 32) uint16 SemanticKITTI labels

    Returns:
        labels: (N,) uint16, 0 = unlabeled for out-of-range
    """
    indices = world_to_voxel_index(points)
    vx, vy, vz = indices[:, 0], indices[:, 1], indices[:, 2]
    in_range = (
        (vx >= 0) & (vx < GRID_SHAPE[0]) &
        (vy >= 0) & (vy < GRID_SHAPE[1]) &
        (vz >= 0) & (vz < GRID_SHAPE[2])
    )
    labels = np.zeros(points.shape[0], dtype=np.uint16)
    labels[~in_range] = 0
    labels[in_range] = voxel_labels[vx[in_range], vy[in_range], vz[in_range]]
    return labels


def load_voxel_labels_raw(voxel_label_path: str) -> np.ndarray:
    """Load raw voxel .label file and reshape to (256, 256, 32)."""
    data = np.fromfile(voxel_label_path, dtype=np.uint16)
    return data.reshape(GRID_SHAPE)


def process_frame(
    points_path: str,
    voxel_label_path: str,
    output_label_path: str,
) -> int:
    """
    Assign voxel labels to points and save as SemanticKITTI .label (uint32).
    Returns number of points with valid (in-range) labels.
    """
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    voxel_labels = load_voxel_labels_raw(voxel_label_path)
    labels = assign_labels_from_voxel_grid(points, voxel_labels)
    # SemanticKITTI point labels: uint32, lower 16 bits = semantic id
    out = labels.astype(np.uint32)
    out.tofile(output_label_path)
    return int((labels > 0).sum())
