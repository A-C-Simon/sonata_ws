#!/usr/bin/env python3
"""Run VoxFormer label preprocessing on SemanticKITTI."""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from label.label_preprocess import main
from paths_config import DEFAULT_KITTI_ROOT, get_preprocess_root

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", "-r", type=str, default=DEFAULT_KITTI_ROOT,
                        help="SemanticKITTI root (default: from paths_config)")
    parser.add_argument("--kitti_preprocess_root", "-p", type=str, default=get_preprocess_root(),
                        help="Output root for preprocessed labels (default: from paths_config)")
    args = parser.parse_args()
    main(args)
