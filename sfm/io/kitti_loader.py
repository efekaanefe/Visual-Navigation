import glob
import os

import numpy as np


class KittiSequenceLoader:
    """Loads image paths, intrinsics and ground-truth positions for one sequence.

    Expects the standard KITTI-odometry layout used by the TSformer-VO project::

        <data_root>/<sequence>/image_<camera_id>/*.png
        <data_root>/<sequence>/calib.txt        # lines 'P0:' .. 'P3:'
        <poses_dir>/<sequence>.txt               # 12-value 3x4 rows (optional)
    """

    def __init__(self, data_root, sequence, camera_id, poses_dir):
        self._data_root = data_root
        self._sequence = sequence
        self._camera_id = camera_id
        self._poses_dir = poses_dir

    def image_paths(self):
        pattern = os.path.join(
            self._data_root, self._sequence, "image_{}".format(self._camera_id), "*.png"
        )
        return sorted(glob.glob(pattern))

    def intrinsic_matrix(self):
        calib_path = os.path.join(self._data_root, self._sequence, "calib.txt")
        projection_matrix = self._read_projection_matrix(calib_path, self._camera_id)
        return projection_matrix[:3, :3]

    def ground_truth_positions(self):
        poses_path = os.path.join(self._poses_dir, "{}.txt".format(self._sequence))
        if not os.path.exists(poses_path):
            return None
        positions = []
        with open(poses_path, "r") as poses_file:
            for line in poses_file:
                values = np.array(line.split(), dtype=np.float64)
                positions.append(values.reshape(3, 4)[:, 3])
        return np.array(positions)

    @staticmethod
    def _read_projection_matrix(calib_path, camera_id):
        with open(calib_path, "r") as calib_file:
            lines = calib_file.readlines()
        tokens = lines[int(camera_id)].strip().split()
        if tokens[0].endswith(":"):
            tokens = tokens[1:]
        return np.array(tokens, dtype=np.float64).reshape(3, 4)
