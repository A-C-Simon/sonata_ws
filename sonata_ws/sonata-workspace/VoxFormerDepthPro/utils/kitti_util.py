"""KITTI calibration and projection utilities (from VoxFormer)."""

from __future__ import print_function

import numpy as np


class Calibration:
    """Calibration for KITTI. Uses P2 and Tr."""

    def __init__(self, calib_filepath):
        calibs = self._read_calib_file(calib_filepath)
        self.P = calibs.get("P2", calibs.get("P", None))
        if self.P is None:
            raise ValueError("P2 or P not found in calib")
        self.P = np.reshape(self.P, [3, 4])
        self.V2C = calibs.get("Tr", calibs.get("Tr_velo_to_cam", None))
        if self.V2C is None:
            raise ValueError("Tr or Tr_velo_to_cam not found")
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = _inverse_rigid_trans(self.V2C)
        self.R0 = np.reshape(np.eye(3), [3, 3])
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)
        self.b_y = self.P[1, 3] / (-self.f_v)

    def _read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                key, val = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in val.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        return np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

    def project_velo_to_ref(self, pts_3d_velo):
        return np.dot(self.cart2hom(pts_3d_velo), np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        return np.dot(self.cart2hom(pts_3d_ref), np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        return self.project_ref_to_velo(self.project_rect_to_ref(pts_3d_rect))

    def project_velo_to_rect(self, pts_3d_velo):
        return self.project_ref_to_rect(self.project_velo_to_ref(pts_3d_velo))

    def project_image_to_rect(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts = np.zeros((n, 3))
        pts[:, 0], pts[:, 1], pts[:, 2] = x, y, uv_depth[:, 2]
        return pts

    def project_image_to_velo(self, uv_depth):
        return self.project_rect_to_velo(self.project_image_to_rect(uv_depth))


def _inverse_rigid_trans(Tr):
    inv = np.zeros_like(Tr)
    inv[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv[0:3, 3] = -np.dot(inv[0:3, 0:3], Tr[0:3, 3])
    return inv
