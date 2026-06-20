from collections import deque

import cv2
import numpy as np

from sfm.core.reconstruction import Reconstruction
from sfm.core.view import View


class IncrementalSfMPipeline:
    """Drives the full incremental SfM pipeline over an image sequence.

    Dependencies (feature extractor, matcher, the per-module stages) are injected
    so the orchestrator only sequences them: initialise the map, then for every
    new frame track its pose, expand the map and periodically run local BA.
    """

    def __init__(self, config, feature_extractor, feature_matcher, relative_pose_estimator,
                 initializer, tracker, map_expander, bundle_adjuster):
        self._config = config
        self._feature_extractor = feature_extractor
        self._feature_matcher = feature_matcher
        self._relative_pose_estimator = relative_pose_estimator
        self._initializer = initializer
        self._tracker = tracker
        self._map_expander = map_expander
        self._bundle_adjuster = bundle_adjuster

    def reconstruct(self, image_paths, intrinsic_matrix, progress=None):
        reconstruction = Reconstruction()
        active_view_ids = deque()

        first_view, second_view_index = self._initialize(reconstruction, image_paths, intrinsic_matrix, progress)
        if first_view is None:
            return reconstruction
        self._remember_active_views(reconstruction, active_view_ids)

        previous_view = reconstruction.ordered_views()[-1]
        previous_center = previous_view.camera_center()
        recent_steps = deque(maxlen=self._config.step_history_size)
        frames_since_local_ba = 0
        for frame_index in range(second_view_index + 1, len(image_paths)):
            current_view = self._build_view(frame_index, image_paths[frame_index], intrinsic_matrix)
            reconstruction.add_view(current_view)
            tracked = self._tracker.track(reconstruction, current_view)
            if tracked and self._is_implausible_step(current_view, previous_center, recent_steps):
                tracked = False
            if not tracked:
                tracked = self._recover(reconstruction, previous_view, current_view, recent_steps)

            if tracked:
                recent_steps.append(float(np.linalg.norm(current_view.camera_center() - previous_center)))
                reference_view = self._reference_view(reconstruction, current_view)
                index_pairs = self._feature_matcher.match(reference_view.descriptors, current_view.descriptors)
                self._map_expander.expand(reconstruction, reference_view, current_view, index_pairs)
                frames_since_local_ba = self._maybe_run_local_ba(reconstruction, frames_since_local_ba)
            else:
                current_view.set_pose(previous_view.rotation, previous_view.translation)

            previous_center = current_view.camera_center()
            previous_view = current_view
            self._track_active_view(reconstruction, active_view_ids, current_view.id)
            if progress is not None:
                progress.update(1)
        return reconstruction

    def _recover(self, reconstruction, previous_view, current_view, recent_steps):
        """Re-localise a lost frame from a two-view relative pose (essential matrix).

        When map-based tracking fails, the local map has starved.  A relative pose
        between the previous and current frames almost always succeeds, so we chain
        it onto the previous absolute pose with a constant-velocity scale (the recent
        median step).  The normal expansion step then refills the map, breaking the
        permanent-freeze death spiral that map-only tracking suffers on long drives.
        """
        if previous_view.descriptors is None or current_view.descriptors is None:
            return False
        index_pairs = self._feature_matcher.match(previous_view.descriptors, current_view.descriptors)
        if len(index_pairs) < self._config.min_pnp_correspondences:
            return False

        pixels_previous = np.array([previous_view.keypoints[int(i)].pt for i in index_pairs[:, 0]], dtype=np.float64)
        pixels_current = np.array([current_view.keypoints[int(i)].pt for i in index_pairs[:, 1]], dtype=np.float64)
        success, rotation, translation, _ = self._relative_pose_estimator.estimate(
            pixels_previous, pixels_current, current_view.intrinsic_matrix
        )
        if not success:
            return False

        step = float(np.median(recent_steps)) if len(recent_steps) > 0 else 1.0
        new_rotation = rotation @ previous_view.rotation
        new_translation = rotation @ previous_view.translation + translation * step
        current_view.set_pose(new_rotation, new_translation)
        return True

    def _is_implausible_step(self, current_view, previous_center, recent_steps):
        """Reject a pose whose motion dwarfs (jump) or collapses below (creep/freeze) the median step."""
        if len(recent_steps) < self._config.min_steps_for_outlier_check:
            return False
        median_step = float(np.median(recent_steps))
        if median_step < 1e-6:
            return False
        step = float(np.linalg.norm(current_view.camera_center() - previous_center))
        too_large = step > self._config.max_step_ratio * median_step
        too_small = step < self._config.min_step_ratio * median_step
        return too_large or too_small

    def _initialize(self, reconstruction, image_paths, intrinsic_matrix, progress):
        if len(image_paths) < 2:
            return None, 0
        first_view = self._build_view(0, image_paths[0], intrinsic_matrix)
        last_candidate = min(len(image_paths), self._config.max_initialization_skip + 1)
        for second_view_index in range(1, last_candidate):
            second_view = self._build_view(second_view_index, image_paths[second_view_index], intrinsic_matrix)
            if self._initializer.initialize(reconstruction, first_view, second_view):
                if progress is not None:
                    progress.update(second_view_index + 1)
                return first_view, second_view_index
        return None, 0

    def _maybe_run_local_ba(self, reconstruction, frames_since_local_ba):
        if self._config.local_ba_every <= 0:
            return frames_since_local_ba
        frames_since_local_ba += 1
        if frames_since_local_ba >= self._config.local_ba_every:
            self._bundle_adjuster.refine_local_window(reconstruction)
            return 0
        return frames_since_local_ba

    def _reference_view(self, reconstruction, current_view):
        """A view ~keyframe_gap frames back (with features) for wide-baseline triangulation.

        A multi-frame baseline gives forward-scene points enough parallax to
        triangulate stably, which keeps the local map populated as the car advances.
        """
        for offset in range(self._config.keyframe_gap, 0, -1):
            candidate = reconstruction.views.get(current_view.id - offset)
            if candidate is not None and candidate.descriptors is not None:
                return candidate
        return reconstruction.views[current_view.id - 1]

    def _build_view(self, frame_index, image_path, intrinsic_matrix):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self._feature_extractor.detect_and_compute(gray_image)
        return View(frame_index, keypoints, descriptors, intrinsic_matrix)

    def _track_active_view(self, reconstruction, active_view_ids, view_id):
        active_view_ids.append(view_id)
        while len(active_view_ids) > self._config.feature_retention():
            stale_view_id = active_view_ids.popleft()
            reconstruction.views[stale_view_id].release_features()

    @staticmethod
    def _remember_active_views(reconstruction, active_view_ids):
        for view in reconstruction.ordered_views():
            active_view_ids.append(view.id)
