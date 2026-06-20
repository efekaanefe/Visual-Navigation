import numpy as np


def project_points(points_world, rotation, translation, intrinsic_matrix):
    """Project ``Nx3`` world points to ``Nx2`` pixels with ``x ~ K (R X + t)``.

    Returns ``(pixels, depths)``; ``depths`` lets the caller reject points that
    fall behind the camera without raising.
    """
    points_camera = points_world @ rotation.T + translation.reshape(3)
    depths = points_camera[:, 2]
    safe_depths = np.where(depths == 0.0, 1e-12, depths)
    normalized = points_camera[:, :2] / safe_depths[:, None]
    focal = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    principal_point = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    pixels = normalized * focal + principal_point
    return pixels, depths


def reprojection_errors(points_world, pixels_observed, rotation, translation, intrinsic_matrix):
    pixels_projected, depths = project_points(points_world, rotation, translation, intrinsic_matrix)
    errors = np.linalg.norm(pixels_projected - pixels_observed, axis=1)
    return errors, depths
