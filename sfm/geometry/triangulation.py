import cv2
import numpy as np

from sfm.geometry.projection import reprojection_errors


class Triangulator:
    """DLT triangulation followed by geometric validity gating.

    A triangulated point is accepted only when it lies in front of both cameras,
    within a sensible depth range, reprojects accurately and was seen under enough
    parallax (small parallax gives numerically unstable depth).
    """

    def __init__(self, min_angle_deg, max_reprojection_error, min_depth, max_depth):
        self._min_angle_cosine = np.cos(np.deg2rad(min_angle_deg))
        self._max_reprojection_error = max_reprojection_error
        self._min_depth = min_depth
        self._max_depth = max_depth

    def triangulate(self, view_a, view_b, pixels_a, pixels_b):
        """Return ``(points_world Nx3, valid_mask)`` for matched pixel pairs."""
        if len(pixels_a) == 0:
            return np.empty((0, 3)), np.zeros(0, dtype=bool)

        homogeneous_points = cv2.triangulatePoints(
            view_a.projection_matrix(), view_b.projection_matrix(),
            pixels_a.T, pixels_b.T,
        )
        points_world = (homogeneous_points[:3] / homogeneous_points[3]).T
        valid_mask = self._validity_mask(view_a, view_b, points_world, pixels_a, pixels_b)
        return points_world, valid_mask

    def _validity_mask(self, view_a, view_b, points_world, pixels_a, pixels_b):
        errors_a, depths_a = reprojection_errors(
            points_world, pixels_a, view_a.rotation, view_a.translation, view_a.intrinsic_matrix
        )
        errors_b, depths_b = reprojection_errors(
            points_world, pixels_b, view_b.rotation, view_b.translation, view_b.intrinsic_matrix
        )
        in_front = (depths_a > self._min_depth) & (depths_b > self._min_depth)
        within_range = (depths_a < self._max_depth) & (depths_b < self._max_depth)
        accurate = (errors_a < self._max_reprojection_error) & (errors_b < self._max_reprojection_error)
        enough_parallax = self._parallax_mask(view_a, view_b, points_world)
        return in_front & within_range & accurate & enough_parallax

    def _parallax_mask(self, view_a, view_b, points_world):
        rays_a = self._normalized_rays(points_world, view_a.camera_center())
        rays_b = self._normalized_rays(points_world, view_b.camera_center())
        cosine_angle = np.sum(rays_a * rays_b, axis=1)
        return cosine_angle < self._min_angle_cosine

    @staticmethod
    def _normalized_rays(points_world, camera_center):
        rays = points_world - camera_center
        norms = np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        return rays / norms
