import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm.geometry.projection import project_points
from sfm.geometry.transforms import rotation_to_vector, vector_to_rotation


class BundleAdjuster:
    """Module 5 - reprojection-error minimisation with ``scipy.least_squares``.

    Two entry points share the same machinery:
      - :meth:`refine_single_view` is motion-only BA (used by the tracker): it
        optimises one camera pose against fixed landmarks.
      - :meth:`refine_local_window` is local BA over the most recent views and the
        landmarks they share, with the oldest views held fixed to remove the gauge
        freedom (including monocular scale).  A sparse Jacobian keeps it tractable.
    """

    def __init__(self, config):
        self._config = config

    # ------------------------------------------------------------------ #
    # Motion-only bundle adjustment (single pose, fixed structure)
    # ------------------------------------------------------------------ #
    def refine_single_view(self, reconstruction, view):
        points_world, pixels = self._view_observations(reconstruction, view)
        if len(points_world) < self._config.min_pnp_correspondences:
            return
        initial_parameters = self._pack_pose(view.rotation, view.translation)
        result = least_squares(
            self._single_view_residuals, initial_parameters,
            args=(points_world, pixels, view.intrinsic_matrix),
            method="trf", loss=self._config.bundle_adjustment_loss,
            f_scale=self._config.bundle_adjustment_f_scale,
            max_nfev=self._config.bundle_adjustment_max_iterations,
        )
        rotation, translation = self._unpack_pose(result.x)
        view.set_pose(rotation, translation)

    @staticmethod
    def _single_view_residuals(parameters, points_world, pixels, intrinsic_matrix):
        rotation = vector_to_rotation(parameters[:3])
        translation = parameters[3:6]
        projected, _ = project_points(points_world, rotation, translation, intrinsic_matrix)
        return (projected - pixels).ravel()

    # ------------------------------------------------------------------ #
    # Local window bundle adjustment (recent poses + shared structure)
    # ------------------------------------------------------------------ #
    def refine_local_window(self, reconstruction):
        if self._config.local_ba_every <= 0:
            return
        window_views = reconstruction.recent_views(self._config.local_ba_window_size)
        if len(window_views) < 3:
            return
        built_problem = self._build_local_problem(reconstruction, window_views)
        if built_problem is None:
            return
        problem, variable_views, landmark_ids = built_problem
        result = least_squares(
            self._local_residuals, problem.initial_parameters,
            jac_sparsity=problem.sparsity, args=(problem,),
            method="trf", loss=self._config.bundle_adjustment_loss,
            f_scale=self._config.bundle_adjustment_f_scale,
            max_nfev=self._config.bundle_adjustment_max_iterations,
        )
        self._write_back(reconstruction, variable_views, landmark_ids, result.x)

    def _build_local_problem(self, reconstruction, window_views):
        fixed_count = min(self._config.local_ba_fixed_views, len(window_views) - 1)
        fixed_views = window_views[:fixed_count]
        variable_views = window_views[fixed_count:]
        if len(variable_views) == 0:
            return None

        variable_slot = {view.id: slot for slot, view in enumerate(variable_views)}
        fixed_slot = {view.id: slot for slot, view in enumerate(fixed_views)}

        landmark_ids = self._window_landmark_ids(reconstruction, window_views)
        if len(landmark_ids) == 0:
            return None
        point_slot = {landmark_id: slot for slot, landmark_id in enumerate(landmark_ids)}

        observations = self._collect_window_observations(
            reconstruction, window_views, variable_slot, fixed_slot, point_slot
        )
        if len(observations.pixel) < self._config.min_pnp_correspondences:
            return None

        initial_parameters = self._pack_local_parameters(reconstruction, variable_views, landmark_ids)
        fixed_rotations = np.array([view.rotation for view in fixed_views]).reshape(-1, 3, 3)
        fixed_translations = np.array([view.translation for view in fixed_views]).reshape(-1, 3)

        problem = LocalBundleProblem(
            variable_views[0].intrinsic_matrix, len(variable_views), len(landmark_ids),
            observations, fixed_rotations, fixed_translations, initial_parameters,
        )
        return problem, variable_views, landmark_ids

    def _collect_window_observations(self, reconstruction, window_views, variable_slot, fixed_slot, point_slot):
        is_variable = []
        view_slot = []
        point_index = []
        pixel = []
        for view in window_views:
            view_is_variable = view.id in variable_slot
            slot = variable_slot[view.id] if view_is_variable else fixed_slot[view.id]
            for keypoint_index, landmark_id in reconstruction.keypoint_to_landmark.get(view.id, {}).items():
                if landmark_id not in point_slot:
                    continue
                is_variable.append(view_is_variable)
                view_slot.append(slot)
                point_index.append(point_slot[landmark_id])
                pixel.append(view.keypoints[keypoint_index].pt)
        return WindowObservations(
            np.array(is_variable, dtype=bool),
            np.array(view_slot, dtype=np.int64),
            np.array(point_index, dtype=np.int64),
            np.array(pixel, dtype=np.float64).reshape(-1, 2),
        )

    def _window_landmark_ids(self, reconstruction, window_views):
        observation_counts = {}
        for view in window_views:
            for landmark_id in reconstruction.keypoint_to_landmark.get(view.id, {}).values():
                observation_counts[landmark_id] = observation_counts.get(landmark_id, 0) + 1
        minimum = self._config.min_landmark_window_observations
        return [landmark_id for landmark_id, count in observation_counts.items() if count >= minimum]

    def _pack_local_parameters(self, reconstruction, variable_views, landmark_ids):
        pose_parameters = [self._pack_pose(view.rotation, view.translation) for view in variable_views]
        point_parameters = [reconstruction.landmarks[landmark_id].position for landmark_id in landmark_ids]
        return np.concatenate(pose_parameters + point_parameters)

    @staticmethod
    def _local_residuals(parameters, problem):
        pose_block = parameters[:problem.pose_parameter_count].reshape(problem.variable_view_count, 6)
        point_block = parameters[problem.pose_parameter_count:].reshape(problem.point_count, 3)

        rotations, translations = problem.observation_poses(pose_block)
        points = point_block[problem.observations.point_index]
        points_camera = np.einsum("mij,mj->mi", rotations, points) + translations

        depths = points_camera[:, 2]
        safe_depths = np.where(depths == 0.0, 1e-12, depths)
        projected = points_camera[:, :2] / safe_depths[:, None]
        projected = projected * problem.focal + problem.principal_point
        return (projected - problem.observations.pixel).ravel()

    @staticmethod
    def _write_back(reconstruction, variable_views, landmark_ids, parameters):
        pose_count = len(variable_views)
        pose_block = parameters[:6 * pose_count].reshape(pose_count, 6)
        point_block = parameters[6 * pose_count:].reshape(len(landmark_ids), 3)
        for slot, view in enumerate(variable_views):
            view.set_pose(vector_to_rotation(pose_block[slot, :3]), pose_block[slot, 3:6])
        for slot, landmark_id in enumerate(landmark_ids):
            reconstruction.landmarks[landmark_id].position = point_block[slot]

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _view_observations(reconstruction, view):
        points_world = []
        pixels = []
        for keypoint_index, landmark_id in reconstruction.keypoint_to_landmark.get(view.id, {}).items():
            points_world.append(reconstruction.landmarks[landmark_id].position)
            pixels.append(view.keypoints[keypoint_index].pt)
        return (
            np.array(points_world, dtype=np.float64).reshape(-1, 3),
            np.array(pixels, dtype=np.float64).reshape(-1, 2),
        )

    @staticmethod
    def _pack_pose(rotation, translation):
        return np.concatenate([rotation_to_vector(rotation), translation.reshape(3)])

    @staticmethod
    def _unpack_pose(parameters):
        return vector_to_rotation(parameters[:3]), parameters[3:6]


class WindowObservations:
    """Flat arrays describing every reprojection residual in a local BA problem."""

    def __init__(self, is_variable, view_slot, point_index, pixel):
        self.is_variable = is_variable      # bool (M,)
        self.view_slot = view_slot          # int  (M,) slot into variable or fixed views
        self.point_index = point_index      # int  (M,) slot into the landmark list
        self.pixel = pixel                  # (M, 2) observed coordinates


class LocalBundleProblem:
    """Immutable description of a local BA problem (parameters + sparsity)."""

    def __init__(self, intrinsic_matrix, variable_view_count, point_count,
                 observations, fixed_rotations, fixed_translations, initial_parameters):
        self.focal = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
        self.principal_point = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
        self.variable_view_count = variable_view_count
        self.point_count = point_count
        self.pose_parameter_count = 6 * variable_view_count
        self.observations = observations
        self.fixed_rotations = fixed_rotations
        self.fixed_translations = fixed_translations
        self.initial_parameters = initial_parameters
        self.sparsity = self._build_sparsity()

    def observation_poses(self, pose_block):
        """Return per-observation ``(rotations Mx3x3, translations Mx3)``."""
        variable_rotations = self._variable_rotations(pose_block)
        observation_count = len(self.observations.pixel)
        rotations = np.empty((observation_count, 3, 3))
        translations = np.empty((observation_count, 3))

        variable_mask = self.observations.is_variable
        fixed_mask = ~variable_mask
        if np.any(variable_mask):
            slots = self.observations.view_slot[variable_mask]
            rotations[variable_mask] = variable_rotations[slots]
            translations[variable_mask] = pose_block[slots, 3:6]
        if np.any(fixed_mask):
            slots = self.observations.view_slot[fixed_mask]
            rotations[fixed_mask] = self.fixed_rotations[slots]
            translations[fixed_mask] = self.fixed_translations[slots]
        return rotations, translations

    def _variable_rotations(self, pose_block):
        if self.variable_view_count == 0:
            return np.zeros((0, 3, 3))
        return np.array([vector_to_rotation(pose_block[slot, :3]) for slot in range(self.variable_view_count)])

    def _build_sparsity(self):
        observation_count = len(self.observations.pixel)
        parameter_count = self.pose_parameter_count + 3 * self.point_count
        sparsity = lil_matrix((2 * observation_count, parameter_count), dtype=int)
        for observation_index in range(observation_count):
            row = 2 * observation_index
            point_column = self.pose_parameter_count + 3 * self.observations.point_index[observation_index]
            sparsity[row:row + 2, point_column:point_column + 3] = 1
            if self.observations.is_variable[observation_index]:
                pose_column = 6 * self.observations.view_slot[observation_index]
                sparsity[row:row + 2, pose_column:pose_column + 6] = 1
        return sparsity
