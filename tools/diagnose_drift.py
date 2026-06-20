"""Throwaway diagnostic: is the poor trajectory an axis/convention bug or scale drift?

Aligns only the FIRST k estimated camera centres to ground truth (and a few sliding
windows) and prints the per-window similarity scale.  If early windows align tightly
with scale ~ constant, the conventions are correct and the global error is drift.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfm.evaluation.alignment import TrajectoryAligner

POSES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Robotics-Masters-Related", "EE584-Machine_Vision",
    "term_project_repo", "TSformer-VO", "data", "sequences_png", "poses",
)


def camera_centers_from_poses(poses):
    centers = []
    for pose in poses:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        centers.append(-rotation.T @ translation)
    return np.asarray(centers)


def load_gt(sequence):
    path = os.path.normpath(os.path.join(POSES_DIR, "{}.txt".format(sequence)))
    positions = []
    with open(path) as gt_file:
        for line in gt_file:
            values = np.array(line.split(), dtype=np.float64)
            positions.append(values.reshape(3, 4)[:, 3])
    return np.asarray(positions)


def window_ate_and_scale(aligner, estimate, gt, start, length):
    end = min(start + length, len(estimate), len(gt))
    source = estimate[start:end]
    target = gt[start:end]
    aligned, scale = aligner.align(source, target)
    ate = float(np.linalg.norm(aligned - target, axis=1).mean())
    return ate, scale


def main():
    sequence = sys.argv[1] if len(sys.argv) > 1 else "09"
    poses = np.load(os.path.join("results", "classical_vo", "pred_poses_{}.npy".format(sequence)))
    estimate = camera_centers_from_poses(poses)
    gt = load_gt(sequence)
    aligner = TrajectoryAligner()

    print("seq {}: {} estimated poses, {} gt poses".format(sequence, len(estimate), len(gt)))
    full_ate, full_scale = window_ate_and_scale(aligner, estimate, gt, 0, len(estimate))
    print("FULL   align -> ATE {:8.2f} m   scale {:8.3f}".format(full_ate, full_scale))
    for start in range(0, len(estimate) - 50, 200):
        ate, scale = window_ate_and_scale(aligner, estimate, gt, start, 150)
        print("window [{:4d}:{:4d}] ATE {:8.3f} m   scale {:8.3f}".format(start, start + 150, ate, scale))

    count = min(len(estimate), len(gt))
    estimate_steps = np.linalg.norm(np.diff(estimate[:count], axis=0), axis=1)
    gt_steps = np.linalg.norm(np.diff(gt[:count], axis=0), axis=1)
    global_scale = full_scale
    scaled_estimate_steps = estimate_steps * global_scale
    frozen = np.flatnonzero(scaled_estimate_steps < 0.2 * gt_steps)
    jumpy = np.flatnonzero(scaled_estimate_steps > 5.0 * gt_steps)
    print("\nper-frame step (after global scale {:.2f}):".format(global_scale))
    print("  frozen frames (est<<gt): {}  e.g. {}".format(len(frozen), frozen[:15].tolist()))
    print("  jumpy  frames (est>>gt): {}  e.g. {}".format(len(jumpy), jumpy[:15].tolist()))
    print("  gt mean step {:.3f} m, est mean step (scaled) {:.3f} m".format(
        gt_steps.mean(), scaled_estimate_steps.mean()))


if __name__ == "__main__":
    main()
