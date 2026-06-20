import cv2
import numpy as np


class RelativePoseEstimator:
    """Two-view relative pose from the fundamental / essential matrix.

    Follows the classical chain: robust 8-point ``F`` -> ``E = K^T F K`` ->
    decomposition into four ``[R | t]`` candidates -> cheirality check (handled by
    :func:`cv2.recoverPose`, which keeps the solution with the most points in
    front of both cameras).

    The fundamental matrix is estimated with MAGSAC++ (``cv2.USAC_MAGSAC``)
    rather than vanilla ``FM_RANSAC``: besides being a more accurate robust
    estimator, ``FM_RANSAC`` aborts with an internal assertion on some degenerate
    point sets, whereas the USAC path returns ``None`` on failure, keeping the
    pipeline exception-free.
    """

    def __init__(self, ransac_confidence, ransac_pixel_threshold):
        self._confidence = ransac_confidence
        self._threshold = ransac_pixel_threshold

    def estimate(self, points1, points2, intrinsic_matrix):
        """Return ``(success, rotation, translation, inlier_mask)``.

        ``rotation`` / ``translation`` map a point in camera-1 coordinates into
        camera-2 coordinates; ``translation`` has unit norm (monocular scale).
        ``inlier_mask`` is a boolean array over the input correspondences.
        """
        if len(points1) < 8:
            return False, None, None, None

        fundamental, fundamental_mask = cv2.findFundamentalMat(
            points1, points2, cv2.USAC_MAGSAC, self._threshold, self._confidence
        )
        if fundamental is None or fundamental.shape != (3, 3):
            return False, None, None, None

        essential = intrinsic_matrix.T @ fundamental @ intrinsic_matrix
        inlier_mask = fundamental_mask.ravel().astype(bool)
        if np.count_nonzero(inlier_mask) < 8:
            return False, None, None, None

        inlier_points1 = points1[inlier_mask]
        inlier_points2 = points2[inlier_mask]
        pose_inlier_count, rotation, translation, cheirality_mask = cv2.recoverPose(
            essential, inlier_points1, inlier_points2, intrinsic_matrix
        )
        if pose_inlier_count < 8:
            return False, None, None, None

        final_mask = self._combine_masks(inlier_mask, cheirality_mask)
        return True, rotation, translation.reshape(3), final_mask

    @staticmethod
    def _combine_masks(inlier_mask, cheirality_mask):
        final_mask = np.zeros(len(inlier_mask), dtype=bool)
        inlier_indices = np.flatnonzero(inlier_mask)
        passed_cheirality = cheirality_mask.ravel().astype(bool)
        final_mask[inlier_indices[passed_cheirality]] = True
        return final_mask
