from scripts.visual_odometry import VisualOdometry
from scripts.visualizer import Visualizer
from scripts.utils import calculate_trajectory_error, load_images, load_projection_matrices
from scripts.depth_estimation import estimate_depth

import os


def run_monocular_visual_odometry(data_dir="data/KITTI/2", is_left_camera = True, n_features=3000):
    vo = VisualOdometry(data_dir, is_left_camera, n_features)
    visualizer = Visualizer()

    est_poses = vo.estimate_trajectory()
    visualizer.plot_trajectory_comparison(est_poses, vo.gt_poses)

    error = calculate_trajectory_error(est_poses, vo.gt_poses)
    visualizer.plot_trajectory_error(error)
    print(f"Average Trajectory Error: {error.mean():.4f} meters")



def run_depth_estimation():
    data_dir = "data/KITTI/1"
    
    images_left = load_images(data_dir, is_left_camera=True)
    images_right = load_images(data_dir, is_left_camera=False)
    
    calib_file = os.path.join(data_dir, "calib.txt")
    calib = load_projection_matrices(calib_file)
    P_left = calib["left"]
    P_right = calib["right"]

    image_index = 20
    depth_map = estimate_depth(images_left[image_index], images_right[image_index], P_left, P_right)

    visualizer = Visualizer()
    visualizer.visualize_depth_map(depth_map)

   

if __name__ == "__main__":
    # run_monocular_visual_odometry()
    run_depth_estimation()

