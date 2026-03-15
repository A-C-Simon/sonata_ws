#!/usr/bin/env python3
"""
Export a .bin point cloud to .ply for viewing on your PC (CloudCompare, MeshLab, etc.).
No GUI or Open3D needed — runs on headless servers.
Usage: python scripts/export_pointcloud_to_ply.py <path.bin> [output.ply]
"""
import os
import sys
import numpy as np


def load_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32)
    n = len(pts) // 4
    pts = pts.reshape(n, 4)
    return pts[:, :3]


def write_ply(path: str, points: np.ndarray) -> None:
    n = points.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"element vertex %d\n" % n)
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"end_header\n")
        points.astype(np.float32).tofile(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_pointcloud_to_ply.py <path.bin> [output.ply]")
        sys.exit(1)
    bin_path = sys.argv[1]
    if not os.path.isfile(bin_path):
        print("File not found:", bin_path)
        sys.exit(1)
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(bin_path)[0] + ".ply"
    points = load_bin(bin_path)
    write_ply(out_path, points)
    print("Saved %d points to %s" % (len(points), out_path))
    print("Download this file and open in CloudCompare, MeshLab, or Blender.")


if __name__ == "__main__":
    main()
