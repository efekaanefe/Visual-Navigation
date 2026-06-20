"""Classical incremental-SfM visual odometry on KITTI.

Mirrors the TSformer-VO inference CLI so the classical and deep-learning methods
can be compared side by side::

    python run_inference.py --sequences 08 09 10 22

For each sequence the pipeline estimates a monocular (up-to-scale) trajectory,
aligns it to the ground truth with a similarity transform and writes the same set
of artefacts as TSformer-VO under ``results/classical_vo/``:
``trajectory_<seq>.png``, ``pred_poses_<seq>.npy``, ``pred_poses/<seq>.txt`` and
``timing.json``.
"""

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from sfm.config import SfMConfig
from sfm.evaluation.alignment import TrajectoryAligner
from sfm.features.feature_extractor import SiftFeatureExtractor
from sfm.features.feature_matcher import FeatureMatcher
from sfm.geometry.epipolar import RelativePoseEstimator
from sfm.geometry.pnp import PnPPoseEstimator
from sfm.geometry.triangulation import Triangulator
from sfm.io.kitti_loader import KittiSequenceLoader
from sfm.io.results_writer import ResultsWriter
from sfm.pipeline.bundle_adjuster import BundleAdjuster
from sfm.pipeline.initializer import MapInitializer
from sfm.pipeline.map_expander import MapExpander
from sfm.pipeline.sfm_pipeline import IncrementalSfMPipeline
from sfm.pipeline.tracker import FrameTracker
from sfm.visualization.trajectory_plotter import TrajectoryPlotter


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "Robotics-Masters-Related", "EE584-Machine_Vision",
    "term_project_repo", "TSformer-VO", "data", "sequences_png",
))


def apply_bundle_adjustment_setting(config, enabled):
    """Turn both bundle-adjustment stages on or off.

    Disabling leaves the raw PnP / essential-matrix poses unrefined: motion-only BA
    (per frame) and local sliding-window BA (Module 5) are both switched off.
    """
    config.motion_only_ba = enabled
    config.local_ba_every = config.local_ba_every if enabled else 0


def build_pipeline(config):
    feature_extractor = SiftFeatureExtractor(config.max_features)
    feature_matcher = FeatureMatcher(config.lowe_ratio)
    relative_pose_estimator = RelativePoseEstimator(config.ransac_confidence, config.ransac_pixel_threshold)
    triangulator = Triangulator(
        config.min_triangulation_angle_deg, config.max_reprojection_error_px,
        config.min_landmark_depth, config.max_landmark_depth,
    )
    pnp_pose_estimator = PnPPoseEstimator(
        config.pnp_reprojection_threshold_px, config.pnp_confidence, config.pnp_iterations
    )
    bundle_adjuster = BundleAdjuster(config)
    initializer = MapInitializer(feature_matcher, relative_pose_estimator, triangulator, config)
    tracker = FrameTracker(feature_matcher, pnp_pose_estimator, bundle_adjuster, config)
    map_expander = MapExpander(triangulator, config)
    return IncrementalSfMPipeline(
        config, feature_extractor, feature_matcher, relative_pose_estimator,
        initializer, tracker, map_expander, bundle_adjuster
    )


def select_frames(image_paths, stride, max_frames):
    selected_paths = image_paths[::stride]
    if max_frames > 0:
        selected_paths = selected_paths[:max_frames]
    return selected_paths


def selected_ground_truth(ground_truth, selected_count, stride):
    if ground_truth is None:
        return None
    return ground_truth[::stride][:selected_count]


def align_estimate(aligner, positions, reference_positions, view_ids):
    if reference_positions is None:
        return positions, 1.0, None
    valid_ids = [view_id for view_id in view_ids if view_id < len(reference_positions)]
    target_positions = reference_positions[valid_ids]
    source_positions = positions[:len(target_positions)]
    aligned_positions, scale = aligner.align(source_positions, target_positions)
    return aligned_positions, scale, target_positions


def anchor_to_reference(aligned_positions, reference_positions):
    """Shift the aligned estimate so its first point coincides with the GT start.

    Umeyama returns the least-squares-optimal translation (it aligns centroids),
    which leaves the start offset from the origin.  For the comparison plot we add a
    single constant so both trajectories visibly leave the same point; this changes
    position only, not scale or shape, and the ATE metric is left on the unshifted
    Sim(3)-optimal alignment.
    """
    if reference_positions is None or len(aligned_positions) == 0:
        return aligned_positions
    return aligned_positions + (reference_positions[0] - aligned_positions[0])


def absolute_trajectory_error(aligned_positions, target_positions):
    if target_positions is None:
        return None
    count = min(len(aligned_positions), len(target_positions))
    differences = aligned_positions[:count] - target_positions[:count]
    return float(np.linalg.norm(differences, axis=1).mean())


def run_sequence(sequence, loader, pipeline, aligner, writer, plotter, stride, max_frames):
    image_paths = loader.image_paths()
    if len(image_paths) == 0:
        print("  [skip] no frames found for sequence {}".format(sequence))
        return

    selected_paths = select_frames(image_paths, stride, max_frames)
    intrinsic_matrix = loader.intrinsic_matrix()

    progress = tqdm(total=len(selected_paths), desc="Seq {}".format(sequence), unit="frame")
    start_time = time.perf_counter()
    reconstruction = pipeline.reconstruct(selected_paths, intrinsic_matrix, progress)
    elapsed_seconds = time.perf_counter() - start_time
    progress.close()

    view_ids = sorted(reconstruction.views)
    if len(view_ids) < 2:
        print("  [skip] initialization failed for sequence {}".format(sequence))
        return

    poses = reconstruction.pose_matrices()
    positions = reconstruction.camera_centers()
    reference_positions = selected_ground_truth(
        loader.ground_truth_positions(), len(selected_paths), stride
    )
    aligned_positions, scale, target_positions = align_estimate(
        aligner, positions, reference_positions, view_ids
    )

    milliseconds_per_frame = elapsed_seconds / len(selected_paths) * 1000.0
    error = absolute_trajectory_error(aligned_positions, target_positions)
    timing_entry = {
        "elapsed_s": round(elapsed_seconds, 3),
        "frames": len(selected_paths),
        "registered_views": len(view_ids),
        "landmarks": len(reconstruction.landmarks),
        "ms_per_frame": round(milliseconds_per_frame, 3),
        "scale_to_gt": round(float(scale), 6),
        "ate_m": None if error is None else round(error, 4),
    }

    writer.save_predicted_poses(sequence, poses)
    writer.save_kitti_trajectory(sequence, poses)
    writer.update_timing(sequence, timing_entry)

    title_suffix = "{:.1f}s ({:.1f} ms/frame)".format(elapsed_seconds, milliseconds_per_frame)
    plot_positions = anchor_to_reference(aligned_positions, reference_positions)
    plot_path = plotter.plot(sequence, plot_positions, reference_positions, title_suffix)

    error_text = "n/a" if error is None else "{:.3f} m".format(error)
    print("  views={} landmarks={} ATE={} scale={:.3f}".format(
        len(view_ids), len(reconstruction.landmarks), error_text, scale))
    print("  saved plot -> {}".format(plot_path))


def main():
    parser = argparse.ArgumentParser(description="Classical incremental-SfM VO on KITTI.")
    parser.add_argument("--sequences", nargs="+", default=["08", "09", "10", "22"])
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--camera_id", default="0")
    parser.add_argument("--poses_dir", default=None)
    parser.add_argument("--output_dir", default=os.path.join("results", "classical_vo"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--bundle-adjustment", dest="bundle_adjustment",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable motion-only and local bundle adjustment (default: on). "
                             "Use --no-bundle-adjustment for a raw PnP/essential-matrix baseline.")
    arguments = parser.parse_args()

    poses_dir = arguments.poses_dir
    if poses_dir is None:
        poses_dir = os.path.join(arguments.data_root, "poses")

    config = SfMConfig()
    apply_bundle_adjustment_setting(config, arguments.bundle_adjustment)
    pipeline = build_pipeline(config)
    print("Bundle adjustment: {}".format("on" if arguments.bundle_adjustment else "off"))
    aligner = TrajectoryAligner()
    writer = ResultsWriter(arguments.output_dir)
    plotter = TrajectoryPlotter(arguments.output_dir)

    print("Data root: {}".format(arguments.data_root))
    for sequence in arguments.sequences:
        print("\nSequence {}".format(sequence))
        loader = KittiSequenceLoader(arguments.data_root, sequence, arguments.camera_id, poses_dir)
        run_sequence(sequence, loader, pipeline, aligner, writer, plotter,
                     arguments.stride, arguments.max_frames)

    print("\nDone.")


if __name__ == "__main__":
    main()
