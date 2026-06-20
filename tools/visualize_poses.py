"""Visualise saved pred_poses_<seq>.npy in different reference frames.

The .npy files are RAW world->camera poses (4x4) produced by run_inference.py.
This tool plots the camera-centre trajectory in one of three views:

  raw       - the estimate in its own (arbitrary monocular) scale, with the
              (unaligned) ground truth overlaid so the scale/orientation gap is clear.
  aligned   - Umeyama similarity (scale + rotation + translation) fit to ground truth.
  anchored  - aligned, then shifted so the start coincides with the ground-truth origin.

Plots are written to a separate folder (default results/classical_vo/views) as
trajectory_<seq>_<mode>.png so the canonical run_inference plots are not overwritten.

    python tools/visualize_poses.py --sequences 09 --mode all
    python tools/visualize_poses.py --sequences 08 09 10 --mode anchored
    python tools/visualize_poses.py --sequences 09 --mode raw
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import DEFAULT_DATA_ROOT, anchor_to_reference
from sfm.evaluation.alignment import TrajectoryAligner
from sfm.io.kitti_loader import KittiSequenceLoader
from sfm.visualization.trajectory_plotter import TrajectoryPlotter

MODES = ("raw", "aligned", "anchored")


def camera_centers_from_poses(poses):
    """Camera centres in the world frame from world->camera extrinsics: C = -R^T t."""
    return np.asarray([-pose[:3, :3].T @ pose[:3, 3] for pose in poses])


def transformed_trajectory(mode, positions, ground_truth, aligner):
    """Return (estimate_to_plot, ground_truth_to_plot, scale) for the requested mode."""
    if ground_truth is None:
        return positions, None, 1.0
    reference = ground_truth[:len(positions)]
    if mode == "raw":
        # Keep the estimate in its own monocular scale, but still overlay the
        # (unaligned) ground truth so the scale/orientation gap is visible.
        return positions, reference, 1.0
    aligned, scale = aligner.align(positions, reference)
    if mode == "anchored":
        aligned = anchor_to_reference(aligned, reference)
    return aligned, reference, scale


def title_suffix(mode, scale):
    if mode == "raw":
        return "raw (up to scale)"
    return "{} (scale {:.2f})".format(mode, scale)


def visualize_sequence(sequence, requested_mode, input_dir, loader, aligner, plotter):
    poses_path = os.path.join(input_dir, "pred_poses_{}.npy".format(sequence))
    if not os.path.exists(poses_path):
        print("  [skip] {} not found".format(poses_path))
        return

    positions = camera_centers_from_poses(np.load(poses_path))
    ground_truth = loader.ground_truth_positions()
    modes = MODES if requested_mode == "all" else (requested_mode,)
    for mode in modes:
        estimate, reference, scale = transformed_trajectory(mode, positions, ground_truth, aligner)
        filename = "trajectory_{}_{}.png".format(sequence, mode)
        path = plotter.plot(sequence, estimate, reference, title_suffix(mode, scale), filename=filename)
        print("  {} [{}] -> {}".format(sequence, mode, path))


def main():
    parser = argparse.ArgumentParser(description="Visualise saved pose .npy files (raw/aligned/anchored).")
    parser.add_argument("--sequences", nargs="+", default=["09"])
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="anchored")
    parser.add_argument("--input_dir", default=os.path.join("results", "classical_vo"))
    parser.add_argument("--output_dir", default=os.path.join("results", "classical_vo", "views"))
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--poses_dir", default=None)
    parser.add_argument("--camera_id", default="0")
    arguments = parser.parse_args()

    poses_dir = arguments.poses_dir
    if poses_dir is None:
        poses_dir = os.path.join(arguments.data_root, "poses")

    aligner = TrajectoryAligner()
    plotter = TrajectoryPlotter(arguments.output_dir)
    for sequence in arguments.sequences:
        loader = KittiSequenceLoader(arguments.data_root, sequence, arguments.camera_id, poses_dir)
        visualize_sequence(sequence, arguments.mode, arguments.input_dir, loader, aligner, plotter)


if __name__ == "__main__":
    main()
