import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrajectoryPlotter:
    """Draws an estimated-vs-ground-truth trajectory in the TSformer-VO style.

    The top-view (x-z plane) plot intentionally matches the deep-learning method's
    figure: solid-blue estimate, dashed-red ground truth, green/red start/end dots,
    equal aspect ratio and the same labelling, so the two can be compared directly.
    """

    def __init__(self, output_dir):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot(self, sequence, estimated_positions, ground_truth_positions, title_suffix, filename=None):
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.plot(estimated_positions[:, 0], estimated_positions[:, 2],
                  "b-", linewidth=1.5, label="Estimated")
        axis.scatter(estimated_positions[0, 0], estimated_positions[0, 2],
                     c="green", s=80, zorder=5, label="Start")
        axis.scatter(estimated_positions[-1, 0], estimated_positions[-1, 2],
                     c="red", s=80, zorder=5, label="End")
        if ground_truth_positions is not None:
            axis.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 2],
                      "r--", linewidth=1.5, label="Ground Truth")

        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.set_title("Classical VO | Seq {} | {}".format(sequence, title_suffix))
        axis.legend()
        axis.grid(True)
        axis.set_aspect("equal")

        if filename is None:
            filename = "trajectory_{}.png".format(sequence)
        path = os.path.join(self._output_dir, filename)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return path
