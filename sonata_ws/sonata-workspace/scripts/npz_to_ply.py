#!/usr/bin/env python3
"""
Convert ground_truth *.npz (key 'points') to PLY for CloudCompare / MeshLab.

  python scripts/npz_to_ply.py --input path/to/000000.npz --output out.ply
  python scripts/npz_to_ply.py --input ground_truth/00 --glob "*.npz" --out_dir ply_out/00
"""

import argparse
import glob
import os
import sys

import numpy as np


def write_ply_ascii_xyz(path: str, pts: np.ndarray) -> None:
    """ASCII PLY (x,y,z only). Opens in CloudCompare; no Open3D / GUI libs."""
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.8f")


def npz_to_ply(npz_path: str, ply_path: str) -> None:
    data = np.load(npz_path)
    if "points" not in data.files:
        raise KeyError(f"{npz_path}: expected key 'points', got {data.files}")
    pts = np.asarray(data["points"], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")

    out_dir = os.path.dirname(ply_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_ply_ascii_xyz(ply_path, pts)


def main():
    p = argparse.ArgumentParser(description="NPZ (points) -> PLY for CloudCompare")
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to one .npz file OR directory with npz files",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .ply path (only if --input is a single file)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for batch mode (directory input)",
    )
    p.add_argument(
        "--glob",
        default="*.npz",
        help="Pattern inside directory (default: *.npz)",
    )
    args = p.parse_args()

    inp = os.path.abspath(args.input)

    if os.path.isfile(inp):
        if not inp.endswith(".npz"):
            print("File should be .npz", file=sys.stderr)
            sys.exit(1)
        out = args.output
        if not out:
            out = os.path.splitext(inp)[0] + ".ply"
        npz_to_ply(inp, out)
        print(f"Wrote {out}")
        return

    if not os.path.isdir(inp):
        print(f"Not a file or directory: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(inp, "_ply")
    pattern = os.path.join(inp, args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}", file=sys.stderr)
        sys.exit(1)

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        ply = os.path.join(out_dir, f"{base}.ply")
        npz_to_ply(f, ply)
        print(f"Wrote {ply}")

    print(f"Done: {len(files)} files -> {out_dir}")


if __name__ == "__main__":
    main()
