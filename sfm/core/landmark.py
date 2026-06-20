import numpy as np


class Landmark:
    """A reconstructed 3D point and the 2D observations that support it."""

    def __init__(self, landmark_id, position, descriptor):
        self.id = landmark_id
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.descriptor = descriptor          # representative descriptor
        self.observations = {}                # view_id -> keypoint index

    def add_observation(self, view_id, keypoint_index):
        self.observations[view_id] = keypoint_index

    def observation_count(self):
        return len(self.observations)
