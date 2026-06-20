import numpy as np


class View:
    """A single camera frame together with its estimated pose.

    The pose is stored as a world-to-camera extrinsic so that a world point ``X``
    projects with ``x ~ K [R | t] X``.  The camera centre in world coordinates is
    therefore ``C = -R^T t``.
    """

    def __init__(self, view_id, keypoints, descriptors, intrinsic_matrix):
        self.id = view_id
        self.keypoints = keypoints                # tuple of cv2.KeyPoint
        self.descriptors = descriptors            # (N, D) float32 array
        self.intrinsic_matrix = intrinsic_matrix  # (3, 3)
        self.rotation = np.eye(3)                 # world -> camera
        self.translation = np.zeros(3)            # world -> camera

    def set_pose(self, rotation, translation):
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)

    def keypoint_pixel(self, keypoint_index):
        return np.array(self.keypoints[keypoint_index].pt, dtype=np.float64)

    def projection_matrix(self):
        extrinsic = np.hstack((self.rotation, self.translation.reshape(3, 1)))
        return self.intrinsic_matrix @ extrinsic

    def camera_center(self):
        return -self.rotation.T @ self.translation

    def pose_matrix(self):
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = self.translation
        return pose

    def release_features(self):
        """Free the large keypoint/descriptor arrays once the view is out of scope."""
        self.keypoints = None
        self.descriptors = None
