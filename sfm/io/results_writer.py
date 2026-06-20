import json
import os

import numpy as np


class ResultsWriter:
    """Persists predictions, KITTI-format poses and accumulating timing info.

    Mirrors the artefacts written by TSformer-VO so the two methods produce
    comparable output folders.
    """

    def __init__(self, output_dir):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_predicted_poses(self, sequence, poses):
        path = os.path.join(self._output_dir, "pred_poses_{}.npy".format(sequence))
        np.save(path, poses)
        return path

    def save_kitti_trajectory(self, sequence, poses):
        directory = os.path.join(self._output_dir, "pred_poses")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "{}.txt".format(sequence))
        with open(path, "w") as trajectory_file:
            for pose in poses:
                row = pose[:3, :].reshape(12)
                trajectory_file.write(" ".join(str(value) for value in row) + "\n")
        return path

    def update_timing(self, sequence, timing_entry):
        path = os.path.join(self._output_dir, "timing.json")
        timing = {}
        if os.path.exists(path):
            with open(path, "r") as timing_file:
                timing = json.load(timing_file)
        timing[sequence] = timing_entry
        with open(path, "w") as timing_file:
            json.dump(timing, timing_file, indent=2)
        return path
