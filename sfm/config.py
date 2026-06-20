from dataclasses import dataclass


@dataclass
class SfMConfig:
    """Every tunable parameter of the incremental SfM pipeline in one place.

    Centralising the constants keeps the algorithm modules free of magic numbers
    and makes experiments reproducible by editing a single object.
    """

    # --- Feature extraction ---
    max_features: int = 3000

    # --- Feature matching (Lowe's ratio test) ---
    lowe_ratio: float = 0.75

    # --- Epipolar geometry (RANSAC fundamental/essential matrix) ---
    ransac_confidence: float = 0.999
    ransac_pixel_threshold: float = 1.0

    # --- Triangulation gating ---
    min_triangulation_angle_deg: float = 0.5
    max_reprojection_error_px: float = 4.0
    min_landmark_depth: float = 0.0
    max_landmark_depth: float = 200.0

    # --- Two-view initialization ---
    min_initialization_matches: int = 50
    min_initialization_landmarks: int = 30
    max_initialization_skip: int = 10

    # --- Incremental tracking (PnP against a persistent local map) ---
    min_pnp_correspondences: int = 12
    local_map_view_count: int = 10
    pnp_reprojection_threshold_px: float = 3.0
    pnp_confidence: float = 0.999
    pnp_iterations: int = 200

    # --- Map expansion / multi-view verification (Module 4) ---
    keyframe_gap: int = 3
    verification_view_count: int = 5
    verification_pixel_radius: float = 4.0
    verification_descriptor_distance: float = 250.0
    min_verification_hits: int = 0

    # --- Track robustness (reject implausibly large jumps or near-zero "creep") ---
    max_step_ratio: float = 8.0
    min_step_ratio: float = 0.1
    step_history_size: int = 20
    min_steps_for_outlier_check: int = 10

    # --- Bundle adjustment (Module 5) ---
    motion_only_ba: bool = True
    local_ba_window_size: int = 6
    local_ba_fixed_views: int = 2
    local_ba_every: int = 5
    min_landmark_window_observations: int = 2
    bundle_adjustment_loss: str = "huber"
    bundle_adjustment_f_scale: float = 2.0
    bundle_adjustment_max_iterations: int = 30

    def feature_retention(self):
        """Number of most-recent views whose keypoints/descriptors stay in memory.

        Older views only need their pose for the trajectory, so their (large)
        feature arrays can be released once they fall outside the windows used by
        keyframe triangulation, verification (Module 4) and local BA (Module 5).
        """
        return max(self.local_ba_window_size, self.verification_view_count, self.keyframe_gap) + 2
