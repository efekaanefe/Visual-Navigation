import numpy as np


class TrajectoryAligner:
    """Similarity (scale + rotation + translation) alignment via Umeyama (1991).

    Monocular SfM recovers the trajectory only up to a global scale, so to compare
    against metric ground truth the single best scale/rotation/translation is
    estimated in closed form and applied to the estimate.
    """

    def align(self, source_points, target_points):
        """Return ``(aligned_source_points, scale)``."""
        if len(source_points) < 3 or len(source_points) != len(target_points):
            return source_points, 1.0
        scale, rotation, translation = self._estimate_similarity(source_points, target_points)
        aligned_points = scale * (source_points @ rotation.T) + translation
        return aligned_points, scale

    @staticmethod
    def _estimate_similarity(source_points, target_points):
        point_count = len(source_points)
        source_mean = source_points.mean(axis=0)
        target_mean = target_points.mean(axis=0)
        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        covariance = (target_centered.T @ source_centered) / point_count
        u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)

        sign_correction = np.eye(3)
        if np.linalg.det(u_matrix) * np.linalg.det(v_transpose) < 0:
            sign_correction[2, 2] = -1.0

        rotation = u_matrix @ sign_correction @ v_transpose
        source_variance = (source_centered ** 2).sum() / point_count
        scale = np.trace(np.diag(singular_values) @ sign_correction) / source_variance
        translation = target_mean - scale * (rotation @ source_mean)
        return scale, rotation, translation
