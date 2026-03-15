#!/usr/bin/env python3
"""Run Depth Pro on SemanticKITTI images and save depth maps (.npy)."""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

def _ensure_depth_pro():
    try:
        import depth_pro
        return depth_pro
    except ImportError as e:
        last_err = e
    # Fallback: use ml-depth-pro from /workspace/ml-depth-pro/src if present
    for candidate in ["/workspace/ml-depth-pro/src", os.path.expanduser("~/ml-depth-pro/src")]:
        if not candidate or not os.path.isdir(candidate):
            continue
        if candidate in sys.path:
            try:
                import depth_pro
                return depth_pro
            except ImportError as e:
                last_err = e
            continue
        sys.path.insert(0, candidate)
        try:
            import depth_pro
            return depth_pro
        except ImportError as e:
            last_err = e
        except Exception as e:
            last_err = e
    raise ImportError(
        "depth_pro not found. Install: cd /workspace/ml-depth-pro && pip install . "
        "Or ensure /workspace/ml-depth-pro/src exists. Last error: %s" % (last_err,)
    ) from last_err

def run_sequence(image_dir, depth_dir, device="cuda"):
    depth_pro = _ensure_depth_pro()
    os.makedirs(depth_dir, exist_ok=True)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device).eval()
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))
    for fn in tqdm(files, desc="Depth Pro"):
        image_path = os.path.join(image_dir, fn)
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        if hasattr(image, "to"):
            image = image.to(device)
        if hasattr(image, "dim") and image.dim() == 3:
            image = image.unsqueeze(0)
        import torch
        with torch.no_grad():
            pred = model.infer(image, f_px=f_px)
        depth = pred["depth"]
        if hasattr(depth, "squeeze"):
            depth = depth.squeeze()
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        out_path = os.path.join(depth_dir, os.path.splitext(fn)[0] + ".npy")
        np.save(out_path, depth.astype(np.float32))

if __name__ == "__main__":
    from paths_config import DEFAULT_KITTI_ROOT, get_depth_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Sequence image folder (default: derived from --kitti_root + --seq)")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Output depth folder (default: derived from --out_root + --seq)")
    parser.add_argument("--kitti_root", type=str, default=DEFAULT_KITTI_ROOT,
                        help="SemanticKITTI root (used if image_dir/depth_dir not set)")
    parser.add_argument("--out_root", type=str, default=get_depth_root(),
                        help="Output root for depth/sequences/XX (used if depth_dir not set)")
    parser.add_argument("--seq", type=str, default="00", help="Sequence ID when using kitti_root/out_root")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.image_dir is None:
        args.image_dir = os.path.join(args.kitti_root, "dataset", "sequences", args.seq, "image_2")
    if args.depth_dir is None:
        args.depth_dir = os.path.join(args.out_root, "sequences", args.seq)
    run_sequence(args.image_dir, args.depth_dir, args.device)
