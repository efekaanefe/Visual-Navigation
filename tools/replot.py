"""Regenerate trajectory plots from saved pred_poses_<seq>.npy (no SfM re-run).

Reuses the pipeline's own aligner, plotter and anchoring so the figures match what
run_inference.py would draw - handy after changing only the plotting/alignment.

    python tools/replot.py 00 01 02 03 04 05 06 07 08 09 10
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import DEFAULT_DATA_ROOT, anchor_to_reference
from sfm.evaluation.alignment import TrajectoryAligner
from sfm.io.kitti_loader import KittiSequenceLoader
from sfm.visualization.trajectory_plotter import TrajectoryPlotter

OUTPUT_DIR = os.path.join("results", "classical_vo")


def camera_centers_from_poses(poses):
    centers = [-pose[:3, :3].T @ pose[:3, 3] for pose in poses]
    return np.asarray(centers)


def title_suffix_for(sequence):
    timing_path = os.path.join(OUTPUT_DIR, "timing.json")
    if not os.path.exists(timing_path):
        return ""
    timing = json.load(open(timing_path))
    entry = timing.get(sequence)
    if entry is None:
        return ""
    return "{:.1f}s ({:.1f} ms/frame)".format(entry["elapsed_s"], entry["ms_per_frame"])


def replot_sequence(sequence, loader, aligner, plotter):
    poses = np.load(os.path.join(OUTPUT_DIR, "pred_poses_{}.npy".format(sequence)))
    positions = camera_centers_from_poses(poses)

    ground_truth = loader.ground_truth_positions()
    reference_positions = None if ground_truth is None else ground_truth[:len(positions)]
    aligned_positions, _ = aligner.align(positions, reference_positions) \
        if reference_positions is not None else (positions, 1.0)
    plot_positions = anchor_to_reference(aligned_positions, reference_positions)

    path = plotter.plot(sequence, plot_positions, reference_positions, title_suffix_for(sequence))
    print("  re-plotted {} -> {}".format(sequence, path))


def main():
    sequences = sys.argv[1:] if len(sys.argv) > 1 else ["08", "09", "10"]
    poses_dir = os.path.join(DEFAULT_DATA_ROOT, "poses")
    aligner = TrajectoryAligner()
    plotter = TrajectoryPlotter(OUTPUT_DIR)
    for sequence in sequences:
        loader = KittiSequenceLoader(DEFAULT_DATA_ROOT, sequence, "0", poses_dir)
        replot_sequence(sequence, loader, aligner, plotter)


if __name__ == "__main__":
    main()
