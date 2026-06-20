import cv2
import numpy as np

from sfm.geometry.transforms import vector_to_rotation


class PnPPoseEstimator:
    """Absolute camera pose from 2D-3D correspondences via PnP inside RANSAC."""

    def __init__(self, reprojection_threshold, confidence, iterations):
        self._reprojection_threshold = reprojection_threshold
        self._confidence = confidence
        self._iterations = iterations

    def estimate(self, points_world, pixels, intrinsic_matrix):
        """Return ``(success, rotation, translation, inlier_indices)``.

        ``rotation`` / ``translation`` form the world-to-camera extrinsic.
        ``inlier_indices`` indexes the input correspondences kept by RANSAC.
        """
        if len(points_world) < 4:
            return False, None, None, None

        object_points = points_world.astype(np.float64).reshape(-1, 1, 3)
        image_points = pixels.astype(np.float64).reshape(-1, 1, 2)
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points, image_points, intrinsic_matrix, None,
            reprojectionError=self._reprojection_threshold,
            confidence=self._confidence,
            iterationsCount=self._iterations,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            return False, None, None, None

        rotation = vector_to_rotation(rotation_vector)
        return True, rotation, translation.reshape(3), inliers.ravel()
