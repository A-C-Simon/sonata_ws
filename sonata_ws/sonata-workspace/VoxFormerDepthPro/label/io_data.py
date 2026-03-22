"""
I/O for SemanticKITTI voxel data. Uses Pillow (from Sonata) instead of imageio.
"""

import numpy as np
import yaml
from PIL import Image


def unpack(compressed):
    """Given a bit encoded voxel grid, make a normal voxel grid out of it."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


def _read_SemKITTI(path, dtype, do_unpack):
    data = np.fromfile(path, dtype=dtype)
    if do_unpack:
        data = unpack(data)
    return data


def _read_label_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False).astype(np.float32)
    return label


def _read_invalid_SemKITTI(path):
    invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return invalid


def _read_occluded_SemKITTI(path):
    occluded = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return occluded


def _read_occupancy_SemKITTI(path):
    occupancy = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
    return occupancy


def _read_rgb_SemKITTI(path):
    """Use Pillow (Sonata dep) instead of imageio."""
    rgb = np.array(Image.open(path))
    return rgb


def _get_remap_lut(config_path):
    """Build remap LUT from semantic-kitti config."""
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    maxkey = max(dataset_config['learning_map'].keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(dataset_config['learning_map'].keys())] = list(
        dataset_config['learning_map'].values()
    )
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0
    return remap_lut
