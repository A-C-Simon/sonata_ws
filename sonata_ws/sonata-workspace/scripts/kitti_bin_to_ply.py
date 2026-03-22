#!/usr/bin/env python3
"""
KITTI velodyne .bin (N×4 float32: x,y,z,intensity) -> ASCII PLY for CloudCompare.

  python scripts/kitti_bin_to_ply.py -i path/to/000000.bin -o out.ply
"""

import argparse
import os
import sys

import numpy as np


def write_ply_ascii_xyz(path: str, pts: np.ndarray) -> None:
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.8f")


def main():
    p = argparse.ArgumentParser(description="KITTI velodyne .bin -> PLY")
    p.add_argument("-i", "--input", required=True, help="Path to .bin")
    p.add_argument("-o", "--output", default=None, help="Output .ply (default: same stem .ply)")
    args = p.parse_args()

    inp = os.path.abspath(args.input)
    if not inp.endswith(".bin") or not os.path.isfile(inp):
        print(f"Not a .bin file: {inp}", file=sys.stderr)
        sys.exit(1)

    out = args.output or (os.path.splitext(inp)[0] + ".ply")
    out = os.path.abspath(out)

    raw = np.fromfile(inp, dtype=np.float32)
    if raw.size % 4 != 0:
        print("File size not divisible by 4 (expected x,y,z,intensity)", file=sys.stderr)
        sys.exit(1)
    pts = raw.reshape(-1, 4)[:, :3]
    write_ply_ascii_xyz(out, pts)
    print(f"Wrote {out} ({pts.shape[0]} points)")


if __name__ == "__main__":
    main()
