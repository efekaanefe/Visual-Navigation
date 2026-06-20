import numpy as np

from sfm.geometry.projection import project_points


class MapExpander:
    """Module 4 - grow the map with new, geometrically verified landmarks.

    Triangulates previous<->current matches that are not yet linked to a landmark,
    then verifies each candidate by reprojecting it into recent views and looking
    for a descriptor match in the projected neighbourhood.  Candidates with at
    least ``min_verification_hits`` supporting views become landmarks.
    """

    def __init__(self, triangulator, config):
        self._triangulator = triangulator
        self._config = config

    def expand(self, reconstruction, previous_view, current_view, index_pairs):
        new_pairs = self._link_continuations_and_collect(
            reconstruction, previous_view, current_view, index_pairs
        )
        if len(new_pairs) == 0:
            return 0

        pixels_previous = _pixels_at(previous_view, new_pairs[:, 0])
        pixels_current = _pixels_at(current_view, new_pairs[:, 1])
        points_world, valid_mask = self._triangulator.triangulate(
            previous_view, current_view, pixels_previous, pixels_current
        )

        added_count = 0
        for pair, point, is_valid in zip(new_pairs, points_world, valid_mask):
            if not is_valid:
                continue
            descriptor = current_view.descriptors[pair[1]]
            if not self._is_verified(reconstruction, current_view, point, descriptor):
                continue
            self._register_landmark(reconstruction, previous_view, current_view, pair, point, descriptor)
            added_count += 1
        return added_count

    def _link_continuations_and_collect(self, reconstruction, previous_view, current_view, index_pairs):
        """Carry existing tracks into the current view; return pairs for new points.

        If a match continues a landmark the local-map tracker did not catch, the new
        observation is linked here.  Only matches whose current keypoint is still
        unmapped become triangulation candidates for fresh structure.
        """
        candidate_pairs = []
        for previous_index, current_index in index_pairs:
            previous_index = int(previous_index)
            current_index = int(current_index)
            if reconstruction.has_landmark_for_keypoint(current_view.id, current_index):
                continue
            existing_landmark = reconstruction.landmark_for_keypoint(previous_view.id, previous_index)
            if existing_landmark is not None:
                reconstruction.link_observation(existing_landmark, current_view.id, current_index)
                existing_landmark.descriptor = current_view.descriptors[current_index]
                continue
            candidate_pairs.append((previous_index, current_index))
        return np.array(candidate_pairs, dtype=np.int64).reshape(-1, 2)

    @staticmethod
    def _register_landmark(reconstruction, previous_view, current_view, pair, point, descriptor):
        landmark = reconstruction.create_landmark(point, descriptor)
        reconstruction.link_observation(landmark, previous_view.id, int(pair[0]))
        reconstruction.link_observation(landmark, current_view.id, int(pair[1]))

    def _is_verified(self, reconstruction, current_view, point, descriptor):
        if self._config.min_verification_hits <= 0:
            return True
        hit_count = 0
        for view in reconstruction.recent_views(self._config.verification_view_count):
            if view.id == current_view.id or view.descriptors is None:
                continue
            if self._has_supporting_feature(view, point, descriptor):
                hit_count += 1
        return hit_count >= self._config.min_verification_hits

    def _has_supporting_feature(self, view, point, descriptor):
        pixels, depths = project_points(point.reshape(1, 3), view.rotation, view.translation, view.intrinsic_matrix)
        if depths[0] <= 0:
            return False
        keypoint_positions = np.array([keypoint.pt for keypoint in view.keypoints])
        distances = np.linalg.norm(keypoint_positions - pixels[0], axis=1)
        nearby_indices = np.flatnonzero(distances < self._config.verification_pixel_radius)
        for index in nearby_indices:
            descriptor_distance = np.linalg.norm(view.descriptors[index] - descriptor)
            if descriptor_distance < self._config.verification_descriptor_distance:
                return True
        return False


def _pixels_at(view, keypoint_indices):
    return np.array([view.keypoints[index].pt for index in keypoint_indices], dtype=np.float64)
