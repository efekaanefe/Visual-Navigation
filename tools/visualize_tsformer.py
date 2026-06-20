"""Visualise TSformer-VO predictions (raw / aligned / anchored) from saved .npy.

TSformer-VO stores per-window *relative* 6-DOF poses with shape ``(N, W-1, 6)``
(W = window size: Model1->2, Model2->3, Model3->4).  Recovering a trajectory is a
two-step process taken verbatim from the TSformer ``run_inference.py``:

  1. ``post_processing`` merges the overlapping per-window predictions into one
     ordered list of relative poses.
  2. ``recover_trajectory`` de-normalises each pose (dataset mean/std), turns the
     three euler angles into a rotation (zyx) and a 4x4 relative transform, and
     accumulates them into absolute camera positions.

The recovered trajectory is then drawn in one of three reference frames, reusing
the classical-VO aligner / plotter so both methods' figures look identical:

  raw       - TSformer's own (already metric-ish) estimate, with unaligned GT.
  aligned   - Umeyama similarity (scale + rotation + translation) fit to GT.
  anchored  - aligned, then shifted so the start coincides with the GT origin.

For the ``aligned`` mode the per-sequence ``scale_to_gt`` and ``ate_m`` are also
written back into each model's ``timing.json`` (mirroring the classical-VO format).

    python tools/visualize_tsformer.py --models Model1 Model2 Model3 \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 --mode all
"""

import argparse
import functools
import json
import os
import queue
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import DEFAULT_DATA_ROOT, anchor_to_reference
from sfm.evaluation.alignment import TrajectoryAligner
from sfm.io.kitti_loader import KittiSequenceLoader
from sfm.visualization.trajectory_plotter import TrajectoryPlotter

MODES = ("raw", "aligned", "anchored")

DEFAULT_TSFORMER_RESULTS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Robotics-Masters-Related",
    "EE584-Machine_Vision", "term_project_repo", "TSformer-VO", "results",
))

# Dataset normalisation constants (identical to TSformer-VO run_inference.py).
MEAN_ANGLES = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
STD_ANGLES = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
MEAN_T = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
STD_T = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])


def euler_zyx_to_rotation(z, y, x):
    """Rotation matrix for intrinsic z-y-x euler angles (matches datasets.utils)."""
    rotation_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rotation_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return functools.reduce(np.dot, [rotation_x, rotation_y, rotation_z])


def post_processing(pred_poses, window_size):
    """Merge overlapping per-window predictions into one ordered relative-pose list."""
    if window_size == 2:
        return pred_poses.squeeze(1)

    pose_count = pred_poses.shape[0]
    window_queue = queue.Queue(window_size - 1)
    index = 0
    poses = []

    while not window_queue.full():
        window_queue.put(pred_poses[index])
        index += 1

    while index < pose_count:
        if index == window_size - 1:
            poses.append(window_queue.queue[0][0])
            poses.append((window_queue.queue[0][1] + window_queue.queue[1][0]) / 2)
            if window_size == 4:
                poses.append((window_queue.queue[0][2] + window_queue.queue[1][1] + window_queue.queue[2][0]) / 3)
        elif index < pose_count - 1:
            if window_size == 3:
                poses.append((window_queue.queue[0][1] + window_queue.queue[1][0]) / 2)
            elif window_size == 4:
                poses.append((window_queue.queue[0][2] + window_queue.queue[1][1] + window_queue.queue[2][0]) / 3)
        else:
            if window_size == 3:
                poses.append(window_queue.queue[1][1])
            elif window_size == 4:
                poses.append((window_queue.queue[1][2] + window_queue.queue[2][1]) / 2)
                poses.append(window_queue.queue[2][2])
            index += 1

        if index < pose_count - 1:
            index += 1
            window_queue.get()
            window_queue.put(pred_poses[index])

    return np.asarray(poses)


def recover_trajectory(poses):
    """Accumulate de-normalised relative poses into absolute camera positions."""
    trajectory = []
    transform = np.eye(4)
    for pose in poses:
        z, y, x = np.multiply(pose[:3], STD_ANGLES) + MEAN_ANGLES
        translation = np.multiply(pose[3:], STD_T) + MEAN_T
        rotation = euler_zyx_to_rotation(z, y, x)
        relative = np.vstack([np.hstack([rotation, translation.reshape(3, 1)]), [0, 0, 0, 1]])
        transform = transform @ relative
        trajectory.append(transform[:3, 3].copy())
    return np.asarray(trajectory)


def trajectory_from_npy(poses_path):
    raw_predictions = np.load(poses_path)
    window_size = raw_predictions.shape[1] + 1
    relative_poses = post_processing(raw_predictions, window_size)
    return recover_trajectory(relative_poses)


def alignment_metrics(positions, ground_truth, aligner):
    """Return ``(aligned_positions, reference, scale, ate)`` over the shared length."""
    count = min(len(positions), len(ground_truth))
    source = positions[:count]
    reference = ground_truth[:count]
    aligned, scale = aligner.align(source, reference)
    ate = float(np.linalg.norm(aligned - reference, axis=1).mean())
    return aligned, reference, scale, ate


def title_suffix(mode, scale, ate):
    if mode == "raw":
        return "raw (TSformer scale)"
    return "{} | scale {:.3f} | ATE {:.2f} m".format(mode, scale, ate)


def write_timing_metrics(results_dir, sequence, scale, ate):
    """Add scale_to_gt / ate_m to this model's timing.json for the sequence."""
    timing_path = os.path.join(results_dir, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    entry = timing.get(sequence, {})
    entry["scale_to_gt"] = round(float(scale), 6)
    entry["ate_m"] = round(float(ate), 4)
    timing[sequence] = entry
    json.dump(timing, open(timing_path, "w"), indent=2)


def visualize_sequence(sequence, requested_mode, results_dir, loader, aligner, plotter):
    poses_path = os.path.join(results_dir, "pred_poses_{}.npy".format(sequence))
    if not os.path.exists(poses_path):
        print("  [skip] {} not found".format(poses_path))
        return

    positions = trajectory_from_npy(poses_path)
    ground_truth = loader.ground_truth_positions()
    modes = MODES if requested_mode == "all" else (requested_mode,)

    for mode in modes:
        estimate, reference, scale, ate = _frame_for_mode(mode, positions, ground_truth, aligner)
        if mode == "aligned" and ground_truth is not None:
            write_timing_metrics(results_dir, sequence, scale, ate)
        filename = "trajectory_{}_{}.png".format(sequence, mode)
        path = plotter.plot(sequence, estimate, reference, title_suffix(mode, scale, ate), filename=filename)
        print("  {} [{}] -> {}".format(sequence, mode, path))


def _frame_for_mode(mode, positions, ground_truth, aligner):
    if ground_truth is None:
        return positions, None, 1.0, 0.0
    if mode == "raw":
        return positions, ground_truth[:len(positions)], 1.0, 0.0
    aligned, reference, scale, ate = alignment_metrics(positions, ground_truth, aligner)
    if mode == "anchored":
        aligned = anchor_to_reference(aligned, reference)
    return aligned, reference, scale, ate


def main():
    parser = argparse.ArgumentParser(description="Visualise TSformer-VO .npy poses (raw/aligned/anchored).")
    parser.add_argument("--models", nargs="+", default=["Model1", "Model2", "Model3"])
    parser.add_argument("--sequences", nargs="+",
                        default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    parser.add_argument("--results_root", default=DEFAULT_TSFORMER_RESULTS)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--poses_dir", default=None)
    parser.add_argument("--camera_id", default="0")
    arguments = parser.parse_args()

    poses_dir = arguments.poses_dir
    if poses_dir is None:
        poses_dir = os.path.join(arguments.data_root, "poses")

    aligner = TrajectoryAligner()
    for model in arguments.models:
        results_dir = os.path.join(arguments.results_root, model)
        plotter = TrajectoryPlotter(os.path.join(results_dir, "views"))
        print("\n{}".format(model))
        for sequence in arguments.sequences:
            loader = KittiSequenceLoader(arguments.data_root, sequence, arguments.camera_id, poses_dir)
            visualize_sequence(sequence, arguments.mode, results_dir, loader, aligner, plotter)


if __name__ == "__main__":
    main()
