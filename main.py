from scripts.visual_odometry import VisualOdometry
from scripts.visualizer import Visualizer
from scripts.utils import calculate_trajectory_error




if __name__ == "__main__":
    vo = VisualOdometry(data_dir="data/KITTI/2", n_features=3000)

    visualizer = Visualizer()

    est_poses = vo.estimate_trajectory()
    visualizer.plot_trajectory_comparison(est_poses, vo.gt_poses)

    error = calculate_trajectory_error(est_poses, vo.gt_poses)
    print(f"Average Trajectory Error: {error.mean():.4f} meters")