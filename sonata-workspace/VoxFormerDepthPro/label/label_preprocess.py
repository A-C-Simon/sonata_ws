"""Label preprocessing for SemanticKITTI voxels (from VoxFormer)."""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

from . import io_data as SemanticKittiIO


def _downsample_label(label, voxel_size, downscale):
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)
    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])
        label_i[:, :, :] = label[
            x * ds:(x + 1) * ds,
            y * ds:(y + 1) * ds,
            z * ds:(z + 1) * ds,
        ]
        label_bin = label_i.flatten()
        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size
        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            mask = np.logical_and(label_bin > 0, label_bin < 255)
            label_i_s = label_bin[np.where(mask)]
            if len(label_i_s) > 0:
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def main(config):
    voxel_size = (256, 256, 32)
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    label_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(label_dir, "semantic-kitti.yaml")
    remap_lut = SemanticKittiIO._get_remap_lut(config_path)
    for sequence in sequences:
        sequence_path = os.path.join(
            config.kitti_root, "dataset", "sequences", sequence
        )
        label_paths = sorted(glob.glob(
            os.path.join(sequence_path, "voxels", "*.label")))
        invalid_paths = sorted(glob.glob(
            os.path.join(sequence_path, "voxels", "*.invalid")))
        out_dir = os.path.join(config.kitti_preprocess_root, "labels", sequence)
        os.makedirs(out_dir, exist_ok=True)
        downscaling = {"1_1": 1, "1_2": 2}
        if len(label_paths) != len(invalid_paths):
            print("Skipping %s: label/invalid count mismatch" % sequence)
            continue
        for i in tqdm(range(len(label_paths)), desc="Seq %s" % sequence):
            frame_id, _ = os.path.splitext(os.path.basename(label_paths[i]))
            LABEL = SemanticKittiIO._read_label_SemKITTI(label_paths[i])
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_paths[i])
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)
            LABEL[np.isclose(INVALID, 1)] = 255
            LABEL = LABEL.reshape(voxel_size)
            for scale in downscaling:
                filename = frame_id + "_" + scale + ".npy"
                label_filename = os.path.join(out_dir, filename)
                if not os.path.exists(label_filename):
                    if scale != "1_1":
                        LABEL_ds = _downsample_label(
                            LABEL, voxel_size, downscaling[scale])
                    else:
                        LABEL_ds = LABEL
                    np.save(label_filename, LABEL_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", "-r", type=str, required=True)
    parser.add_argument("--kitti_preprocess_root", "-p", type=str, required=True)
    config, _ = parser.parse_known_args()
    main(config)
