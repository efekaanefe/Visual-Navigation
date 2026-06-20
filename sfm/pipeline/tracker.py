import numpy as np


class FrameTracker:
    """Module 3 - estimate the pose of a new view against the existing map.

    Localisation is done against a *persistent local map*: the new view's
    descriptors are matched against the landmarks seen in the last few views, not
    only the immediately previous frame.  This yields many more 2D-3D
    correspondences over a longer baseline, which keeps PnP well-conditioned under
    forward motion and avoids the scale jumps of two-frame tracking.  The recovered
    pose is then refined with motion-only bundle adjustment.
    """

    def __init__(self, feature_matcher, pnp_pose_estimator, bundle_adjuster, config):
        self._feature_matcher = feature_matcher
        self._pnp_pose_estimator = pnp_pose_estimator
        self._bundle_adjuster = bundle_adjuster
        self._config = config

    def track(self, reconstruction, current_view):
        """Return ``True`` when the new view is localised against the map."""
        active_landmarks = reconstruction.active_landmarks(self._config.local_map_view_count)
        if len(active_landmarks) < self._config.min_pnp_correspondences:
            return False

        landmark_descriptors = np.array([landmark.descriptor for landmark in active_landmarks], dtype=np.float32)
        index_pairs = self._feature_matcher.match(current_view.descriptors, landmark_descriptors)
        if len(index_pairs) < self._config.min_pnp_correspondences:
            return False

        keypoint_indices = index_pairs[:, 0]
        matched_landmarks = [active_landmarks[landmark_index] for landmark_index in index_pairs[:, 1]]
        points_world = np.array([landmark.position for landmark in matched_landmarks], dtype=np.float64).reshape(-1, 3)
        pixels = np.array([current_view.keypoints[int(index)].pt for index in keypoint_indices], dtype=np.float64).reshape(-1, 2)

        success, rotation, translation, inlier_indices = self._pnp_pose_estimator.estimate(
            points_world, pixels, current_view.intrinsic_matrix
        )
        if not success:
            return False

        current_view.set_pose(rotation, translation)
        self._link_inliers(reconstruction, current_view, matched_landmarks, keypoint_indices, inlier_indices)
        if self._config.motion_only_ba:
            self._bundle_adjuster.refine_single_view(reconstruction, current_view)
        return True

    @staticmethod
    def _link_inliers(reconstruction, current_view, matched_landmarks, keypoint_indices, inlier_indices):
        for index in inlier_indices:
            landmark = matched_landmarks[index]
            keypoint_index = int(keypoint_indices[index])
            reconstruction.link_observation(landmark, current_view.id, keypoint_index)
            landmark.descriptor = current_view.descriptors[keypoint_index]
