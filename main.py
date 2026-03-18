import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import load_data
from scripts.visualizer import Visualizer, poses_to_SE3



if __name__ == "__main__":
    data_name = "shapes_6dof"
    dataset_path = f"data/{data_name}"
    data = load_data(dataset_path, fraction = 0.01)

    print("Shapes 6DOF data loaded successfully!")
    print("Calibration matrix shape:", data["calib"].shape)
    print("Ground truth poses shape:", data["gt_poses"].shape)
    print("IMU data shape:", data["imu_data"].shape)
    print("Events data shape:", data["events_data"].shape)
    print("Images shape:", data["images"].shape)
    print("Image timestamps shape:", data["image_timestamps"].shape)

    vis = Visualizer()
    T = poses_to_SE3(data["gt_poses"])

    vis.plot_trajectory(T)
    vis.plot_imu(data["imu_data"])

    vis.visualize_events_stream(data["events_data"])

    vis.visualize_images_stream(data["images"])
