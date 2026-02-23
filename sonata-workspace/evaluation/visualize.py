"""
Visualization Tools for Scene Completion Results

Visualize point clouds: single, comparison (partial vs complete vs refined),
and batch visualization from results directories.
"""

import os
import argparse
import numpy as np
import open3d as o3d


def load_pointcloud(path: str) -> np.ndarray:
    """Load point cloud from .bin, .pcd, .ply, or .npz."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        pts = np.fromfile(path, dtype=np.float32)
        pts = pts.reshape(-1, 4)
        return pts[:, :3]
    elif ext == ".npz":
        data = np.load(path)
        return data["points"] if "points" in data else data[list(data.keys())[0]]
    elif ext in [".pcd", ".ply"]:
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def filter_pointcloud(
    points: np.ndarray,
    radius: float = -1.0,
    z_min: float = -2.5,
    z_max: float = 10.0,
) -> np.ndarray:
    """Filter points by radius and height."""
    if radius > 0:
        dist = np.linalg.norm(points[:, :2], axis=1)  # xy only
        mask = dist < radius
    else:
        mask = np.ones(points.shape[0], dtype=bool)
    mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def create_pcd(points: np.ndarray, color: tuple = (0.5, 0.5, 0.5)) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud with color."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def visualize_single(
    path: str,
    radius: float = -1.0,
    z_min: float = -2.5,
    z_max: float = 10.0,
    estimate_normals: bool = True,
):
    """Visualize a single point cloud."""
    points = load_pointcloud(path)
    points = filter_pointcloud(points, radius=radius, z_min=z_min, z_max=z_max)
    pcd = create_pcd(points, (0.5, 0.5, 0.5))
    if estimate_normals:
        pcd.estimate_normals()
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=os.path.basename(path),
        width=1280,
        height=720,
    )


def visualize_comparison(
    partial_path: str,
    complete_path: str,
    refined_path: str = None,
    radius: float = -1.0,
    offset: float = 50.0,
):
    """
    Visualize partial (red), complete (green), and optionally refined (blue) side by side.
    """
    partial = load_pointcloud(partial_path)
    complete = load_pointcloud(complete_path)
    partial = filter_pointcloud(partial, radius=radius)
    complete = filter_pointcloud(complete, radius=radius)

    pcd_partial = create_pcd(partial, (1.0, 0.0, 0.0))
    pcd_complete = create_pcd(complete, (0.0, 1.0, 0.0))
    pcd_complete.translate([offset, 0, 0])

    geoms = [pcd_partial, pcd_complete]

    if refined_path and os.path.exists(refined_path):
        refined = load_pointcloud(refined_path)
        refined = filter_pointcloud(refined, radius=radius)
        pcd_refined = create_pcd(refined, (0.0, 0.0, 1.0))
        pcd_refined.translate([2 * offset, 0, 0])
        geoms.append(pcd_refined)
        title = "Partial (Red) | Complete (Green) | Refined (Blue)"
    else:
        title = "Partial (Red) | Complete (Green)"

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1920,
        height=720,
    )


def visualize_comparison_arrays(
    partial: np.ndarray,
    complete: np.ndarray,
    refined: np.ndarray = None,
    offset: float = 50.0,
):
    """Visualize from numpy arrays (e.g., from inference)."""
    pcd_partial = create_pcd(partial, (1.0, 0.0, 0.0))
    pcd_complete = create_pcd(complete, (0.0, 1.0, 0.0))
    pcd_complete.translate([offset, 0, 0])
    geoms = [pcd_partial, pcd_complete]
    if refined is not None:
        pcd_refined = create_pcd(refined, (0.0, 0.0, 1.0))
        pcd_refined.translate([2 * offset, 0, 0])
        geoms.append(pcd_refined)
    o3d.visualization.draw_geometries(
        geoms,
        window_name="Partial (Red) | Complete (Green)" + (" | Refined (Blue)" if refined is not None else ""),
        width=1920,
        height=720,
    )


def visualize_results_dir(
    results_dir: str,
    idx: int = 0,
    radius: float = -1.0,
):
    """
    Visualize from a results directory with structure:
      results_dir/
        partial/  or scans/
        complete/
        refined/  (optional)
    Or: results_dir/000000_partial.bin, 000000_complete.ply, etc.
    """
    # Try structured layout
    for partial_sub in ["partial", "scans", "input"]:
        partial_dir = os.path.join(results_dir, partial_sub)
        complete_dir = os.path.join(results_dir, "complete")
        refined_dir = os.path.join(results_dir, "refined")
        if os.path.isdir(partial_dir) and os.path.isdir(complete_dir):
            files = sorted([f for f in os.listdir(partial_dir) if f.endswith((".bin", ".ply", ".pcd"))])
            if idx >= len(files):
                print(f"Index {idx} out of range (max {len(files)-1})")
                return
            base = os.path.splitext(files[idx])[0]
            partial_path = os.path.join(partial_dir, files[idx])
            complete_path = os.path.join(complete_dir, base + ".ply")
            if not os.path.exists(complete_path):
                complete_path = os.path.join(complete_dir, base + ".bin")
            refined_path = os.path.join(refined_dir, base + ".ply") if os.path.isdir(refined_dir) else None
            if refined_path and not os.path.exists(refined_path):
                refined_path = None
            print(f"Showing: {base}")
            visualize_comparison(partial_path, complete_path, refined_path, radius=radius)
            return

    # Try flat layout: prefix_idx_suffix.ext
    all_files = [f for f in os.listdir(results_dir) if "_" in f]
    partial_files = [f for f in all_files if "partial" in f or "input" in f]
    if partial_files:
        partial_files = sorted(partial_files)[: idx + 1]
        if partial_files:
            partial_path = os.path.join(results_dir, partial_files[-1])
            base = partial_files[-1].split("_")[0]
            complete_path = os.path.join(results_dir, f"{base}_complete.ply")
            if not os.path.exists(complete_path):
                complete_path = os.path.join(results_dir, f"{base}_complete.bin")
            refined_path = os.path.join(results_dir, f"{base}_refined.ply")
            if not os.path.exists(refined_path):
                refined_path = None
            visualize_comparison(partial_path, complete_path, refined_path, radius=radius)
            return

    print("Could not find valid results structure in", results_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize scene completion results")
    sub = parser.add_subparsers(dest="cmd", help="Command")
    # Single
    p_single = sub.add_parser("single", help="Visualize single point cloud")
    p_single.add_argument("path", help="Path to .bin, .pcd, .ply, or .npz")
    p_single.add_argument("--radius", "-r", type=float, default=-1, help="Filter by xy radius (m)")
    p_single.add_argument("--z-min", type=float, default=-2.5)
    p_single.add_argument("--z-max", type=float, default=10.0)
    p_single.add_argument("--no-normals", action="store_true", help="Skip normal estimation")
    # Comparison
    p_comp = sub.add_parser("comparison", help="Partial vs Complete vs Refined")
    p_comp.add_argument("partial", help="Partial/input scan path")
    p_comp.add_argument("complete", help="Completed output path")
    p_comp.add_argument("refined", nargs="?", default=None, help="Optional refined output path")
    p_comp.add_argument("--radius", "-r", type=float, default=-1)
    p_comp.add_argument("--offset", type=float, default=50, help="Offset between clouds")
    # Results dir
    p_dir = sub.add_parser("dir", help="Visualize from results directory")
    p_dir.add_argument("results_dir", help="Path to results directory")
    p_dir.add_argument("--idx", type=int, default=0, help="Sample index")
    p_dir.add_argument("--radius", "-r", type=float, default=-1)

    args = parser.parse_args()
    if args.cmd == "single":
        visualize_single(
            args.path,
            radius=args.radius,
            z_min=args.z_min,
            z_max=args.z_max,
            estimate_normals=not args.no_normals,
        )
    elif args.cmd == "comparison":
        visualize_comparison(
            args.partial, args.complete, args.refined,
            radius=args.radius, offset=args.offset,
        )
    elif args.cmd == "dir":
        visualize_results_dir(args.results_dir, idx=args.idx, radius=args.radius)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
