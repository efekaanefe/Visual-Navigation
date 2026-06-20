import numpy as np


class MapInitializer:
    """Module 2 - bootstrap the map from the first two usable views.

    Places View-1 at the origin ``[I | 0]``, recovers View-2's relative pose from
    epipolar geometry and triangulates the inlier matches into the first landmarks.
    """

    def __init__(self, feature_matcher, relative_pose_estimator, triangulator, config):
        self._feature_matcher = feature_matcher
        self._relative_pose_estimator = relative_pose_estimator
        self._triangulator = triangulator
        self._config = config

    def initialize(self, reconstruction, first_view, second_view):
        """Return ``True`` when the two views yield a valid initial reconstruction."""
        index_pairs = self._feature_matcher.match(first_view.descriptors, second_view.descriptors)
        if len(index_pairs) < self._config.min_initialization_matches:
            return False

        pixels_first = _pixels_at(first_view, index_pairs[:, 0])
        pixels_second = _pixels_at(second_view, index_pairs[:, 1])
        success, rotation, translation, inlier_mask = self._relative_pose_estimator.estimate(
            pixels_first, pixels_second, first_view.intrinsic_matrix
        )
        if not success:
            return False

        first_view.set_pose(np.eye(3), np.zeros(3))
        second_view.set_pose(rotation, translation)
        reconstruction.add_view(first_view)
        reconstruction.add_view(second_view)

        landmark_count = self._triangulate_inliers(
            reconstruction, first_view, second_view, index_pairs[inlier_mask]
        )
        return landmark_count >= self._config.min_initialization_landmarks

    def _triangulate_inliers(self, reconstruction, first_view, second_view, inlier_pairs):
        pixels_first = _pixels_at(first_view, inlier_pairs[:, 0])
        pixels_second = _pixels_at(second_view, inlier_pairs[:, 1])
        points_world, valid_mask = self._triangulator.triangulate(
            first_view, second_view, pixels_first, pixels_second
        )

        landmark_count = 0
        for pair, point, is_valid in zip(inlier_pairs, points_world, valid_mask):
            if not is_valid:
                continue
            descriptor = first_view.descriptors[pair[0]]
            landmark = reconstruction.create_landmark(point, descriptor)
            reconstruction.link_observation(landmark, first_view.id, int(pair[0]))
            reconstruction.link_observation(landmark, second_view.id, int(pair[1]))
            landmark_count += 1
        return landmark_count


def _pixels_at(view, keypoint_indices):
    return np.array([view.keypoints[index].pt for index in keypoint_indices], dtype=np.float64)
