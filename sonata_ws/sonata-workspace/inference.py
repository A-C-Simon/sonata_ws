"""
Inference Script for Scene Completion

Complete a single LiDAR scan using trained Sonata-LiDiff model.
"""

import os
import torch
import numpy as np
import argparse
from typing import Dict
import open3d as o3d

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import (
    SceneCompletionDiffusion,
    QUERY_MAX_RADIUS,
    QUERY_MIN_RADIUS,
)
from models.refinement_net import RefinementNetwork
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for scene completion')
    
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input scan file (.bin or .pcd)'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output', type=str, default='output_completed.ply',
        help='Output file path'
    )
    parser.add_argument(
        '--denoising_steps', type=int, default=50,
        help='Number of denoising steps'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize result with Open3D'
    )
    parser.add_argument(
        '--voxel_size', type=float, default=0.1,
        help='Voxel size for partial scan (match training / boost GT, default 0.1)'
    )
    parser.add_argument(
        '--refinement_ckpt', type=str, default=None,
        help='Optional refinement checkpoint to densify output'
    )
    parser.add_argument(
        '--up_factor', type=int, default=6,
        help='Refinement up_factor (if using refinement)'
    )
    # Query points for scene completion (aligned with map_from_scans_boost)
    parser.add_argument(
        '--no-query-points', action='store_true',
        help='Disable query points (denoise partial only, no occluded regions)'
    )
    parser.add_argument(
        '--query-max-radius', type=float, default=QUERY_MAX_RADIUS,
        help=f'Max radius for query grid (m), default {QUERY_MAX_RADIUS}'
    )
    parser.add_argument(
        '--query-min-radius', type=float, default=QUERY_MIN_RADIUS,
        help=f'Min radius for query grid (m), default {QUERY_MIN_RADIUS}'
    )
    parser.add_argument(
        '--query-voxel-size', type=float, default=0.15,
        help='Query grid voxel size (m)'
    )
    parser.add_argument(
        '--gt-npz',
        type=str,
        required=True,
        help=(
            'GT map NPZ (boost: key points). Same frame as --input. '
            'target_total = K(GT voxel), K_query = target_total - K(partial).'
        ),
    )
    return parser.parse_args()


def load_scan(path: str) -> np.ndarray:
    """Load point cloud from file."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.bin':
        # KITTI binary format
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # x, y, z, intensity
        return scan[:, :3]
    
    elif ext in ['.pcd', '.ply']:
        # Point Cloud Data format
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_pointcloud(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Save point cloud to file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved point cloud to {path}")


def visualize_comparison(
    partial: np.ndarray,
    complete: np.ndarray
):
    """Visualize partial and completed point clouds side by side."""
    # Create point clouds
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial)
    pcd_partial.paint_uniform_color([1.0, 0.0, 0.0])  # Red for partial
    
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(complete)
    pcd_complete.paint_uniform_color([0.0, 1.0, 0.0])  # Green for completed
    
    # Offset for side-by-side view
    offset = np.array([50, 0, 0])
    pcd_complete.translate(offset)
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd_partial, pcd_complete],
        window_name="Scene Completion: Partial (Red) vs Complete (Green)",
        width=1920,
        height=1080
    )


def prepare_input(
    scan: np.ndarray,
    voxel_size: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Prepare scan for model input.
    
    Args:
        scan: Raw point cloud (N, 3)
        voxel_size: Voxel size for downsampling
        
    Returns:
        Dictionary ready for model input
    """
    # LiDAR: use sensor origin (0,0,0), not mean — scene is asymmetric, ego not at centroid.
    center = np.zeros(3, dtype=np.float32)
    scan_centered = scan - center
    
    # Voxelize (simple grid downsampling)
    voxel_coords = np.floor(scan_centered / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(
        voxel_coords, axis=0, return_inverse=True
    )
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2
    
    # Create fake colors from height
    z_norm = (voxel_centers[:, 2] - voxel_centers[:, 2].min()) / \
             (voxel_centers[:, 2].max() - voxel_centers[:, 2].min() + 1e-6)
    colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
    
    # Convert to tensors
    data = {
        'coord': torch.from_numpy(voxel_centers).float(),
        'color': torch.from_numpy(colors).float(),
        'normal': torch.zeros_like(torch.from_numpy(voxel_centers)).float(),
        'grid_coord': torch.from_numpy(voxel_coords).long(),
        'batch': torch.zeros(voxel_centers.shape[0], dtype=torch.long),
    }
    
    return data, center, inverse


def count_voxel_centers(points_centered: np.ndarray, voxel_size: float) -> int:
    """
    Same voxel grid as prepare_input: unique floor(p / voxel_size) cells.
    points_centered: (N, 3), already centered.
    """
    if points_centered.shape[0] == 0:
        return 0
    vc = np.floor(np.asarray(points_centered, dtype=np.float64) / voxel_size).astype(
        np.int32
    )
    return int(np.unique(vc, axis=0).shape[0])


@torch.no_grad()
def complete_scene(
    model: SceneCompletionDiffusion,
    partial_scan: Dict[str, torch.Tensor],
    num_steps: int = 50,
    use_query_points: bool = True,
    **query_kwargs,
) -> np.ndarray:
    """
    Complete scene from partial scan.

    Args:
        model: Trained scene completion model
        partial_scan: Partial scan data
        num_steps: Number of denoising steps
        use_query_points: Add query grid for occluded regions
        **query_kwargs: Passed to model.complete_scene (query_max_radius, etc.)

    Returns:
        Completed point cloud
    """
    model.eval()

    # Move to device
    device = next(model.parameters()).device
    for key in partial_scan:
        if isinstance(partial_scan[key], torch.Tensor):
            partial_scan[key] = partial_scan[key].to(device)

    # Complete scene
    completed = model.complete_scene(
        partial_scan,
        num_steps=num_steps,
        use_query_points=use_query_points,
        **query_kwargs,
    )

    # Convert to numpy
    completed = completed.cpu().numpy()

    return completed


def main():
    args = parse_args()
    
    print(f"\nLoading scan from: {args.input}")
    scan = load_scan(args.input)
    print(f"Loaded {scan.shape[0]} points")
    
    # Prepare input
    print("\nPreparing input...")
    partial_data, center, inverse = prepare_input(
        scan, voxel_size=args.voxel_size
    )
    n_partial = int(partial_data["coord"].shape[0])
    print(f"Voxelized partial: {n_partial} points (voxel_size={args.voxel_size})")

    if not os.path.isfile(args.gt_npz):
        raise FileNotFoundError(f"--gt-npz not found: {args.gt_npz}")
    gt_raw = np.load(args.gt_npz)["points"].astype(np.float32)
    gt_centered = gt_raw - center
    n_gt_vox = count_voxel_centers(gt_centered, args.voxel_size)
    target_total_points = max(n_gt_vox, n_partial)
    n_query = max(0, target_total_points - n_partial)
    print(
        f"GT NPZ: raw {gt_raw.shape[0]} pts → {n_gt_vox} voxels "
        f"(same center & voxel_size as partial)"
    )
    print(
        f"Target total = {target_total_points} "
        f"(K_query = {n_query} = K(GT_vox) - K(partial))"
    )
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    
    # Build model (same as training)
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=True
    )
    
    condition_extractor = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[2, 3, 4],
        fusion_type="attention"
    )
    
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=1000,
        schedule="cosine",
        denoising_steps=args.denoising_steps
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    use_query = not args.no_query_points
    if use_query:
        print(
            f"Completing scene: query from GT voxel budget, "
            f"r=[{args.query_min_radius}, {args.query_max_radius}]m..."
        )
    else:
        print("Completing scene (denoise only, no query points)...")
    completed = complete_scene(
        model,
        partial_data,
        num_steps=args.denoising_steps,
        use_query_points=use_query,
        query_max_radius=args.query_max_radius,
        query_min_radius=args.query_min_radius,
        query_voxel_size=args.query_voxel_size,
        target_total_points=target_total_points,
    )
    
    # Optional refinement
    if args.refinement_ckpt and os.path.exists(args.refinement_ckpt):
        print("Applying refinement network...")
        refinement = RefinementNetwork(up_factor=args.up_factor).cuda()
        ckpt = load_checkpoint(args.refinement_ckpt)
        refinement.load_state_dict(ckpt.get('model_state_dict', ckpt))
        refinement.eval()
        completed_t = torch.from_numpy(completed).float().cuda()
        with torch.no_grad():
            completed = refinement(completed_t).cpu().numpy()
        print(f"Refined to {completed.shape[0]} points")
    
    # Uncenter
    completed = completed + center
    
    print(f"Completed point cloud: {completed.shape[0]} points")
    
    # Save output
    save_pointcloud(args.output, completed)
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing results...")
        visualize_comparison(scan, completed)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
