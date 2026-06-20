import numpy as np

from sfm.core.landmark import Landmark


class Reconstruction:
    """Global map state: registered views, landmarks and the visibility graph.

    The bipartite visibility graph is stored in both directions for O(1) look-ups:
      - each :class:`Landmark` keeps ``{view_id: keypoint_index}``
      - ``keypoint_to_landmark`` keeps ``view_id -> {keypoint_index: landmark_id}``
    """

    def __init__(self):
        self.views = {}                   # view_id -> View
        self.landmarks = {}               # landmark_id -> Landmark
        self.keypoint_to_landmark = {}    # view_id -> {keypoint_index: landmark_id}
        self._view_order = []             # view ids in insertion (temporal) order
        self._next_landmark_id = 0

    def add_view(self, view):
        if view.id not in self.views:
            self._view_order.append(view.id)
        self.views[view.id] = view
        self.keypoint_to_landmark.setdefault(view.id, {})

    def create_landmark(self, position, descriptor):
        landmark = Landmark(self._next_landmark_id, position, descriptor)
        self.landmarks[landmark.id] = landmark
        self._next_landmark_id += 1
        return landmark

    def link_observation(self, landmark, view_id, keypoint_index):
        landmark.add_observation(view_id, keypoint_index)
        self.keypoint_to_landmark.setdefault(view_id, {})[keypoint_index] = landmark.id

    def landmark_for_keypoint(self, view_id, keypoint_index):
        landmark_id = self.keypoint_to_landmark.get(view_id, {}).get(keypoint_index)
        if landmark_id is None:
            return None
        return self.landmarks[landmark_id]

    def has_landmark_for_keypoint(self, view_id, keypoint_index):
        return keypoint_index in self.keypoint_to_landmark.get(view_id, {})

    def ordered_views(self):
        return [self.views[view_id] for view_id in self._view_order]

    def recent_views(self, count):
        recent_ids = self._view_order[-count:]
        return [self.views[view_id] for view_id in recent_ids]

    def active_landmarks(self, recent_view_count):
        """Landmarks observed in the most recent views - the local map for tracking."""
        recent_ids = self._view_order[-recent_view_count:]
        seen_ids = set()
        landmarks = []
        for view_id in recent_ids:
            for landmark_id in self.keypoint_to_landmark.get(view_id, {}).values():
                if landmark_id in seen_ids:
                    continue
                seen_ids.add(landmark_id)
                landmarks.append(self.landmarks[landmark_id])
        return landmarks

    def camera_centers(self):
        return np.array([view.camera_center() for view in self.ordered_views()])

    def pose_matrices(self):
        return np.array([view.pose_matrix() for view in self.ordered_views()])
