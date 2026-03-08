import numpy as np
import cv2
import os


def calculate_trajectory_error(est_poses, gt_poses):
    error = est_poses - gt_poses
    error = np.linalg.norm(error, axis=1)
    error = np.linalg.norm(error, axis=1)
    return error


# --- Transform Data ---
def get_intrinsic(projection_matrix):
    return projection_matrix[:3, :3]

def get_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

#  --- Load Data ---
def load_projection_matrix_for_camera(file_path):
    """Return the i-th calibration matrix (3x4)"""
    index = 0 
    with open(file_path, 'r') as f:
        lines = f.readlines()
    line = lines[index].strip()
    values = np.array(line.split(), dtype=np.float32)
    P = values.reshape(3, 3)  # projection matrix
    return P

def load_2d_array(file_path):
    """Load poses from file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    arrays = []
    for line in lines:
        array = np.array(line.strip().split(), dtype=np.float32)
        arrays.append(array)
    return np.array(arrays)

def load_image_timestamps(file_path):
    """Load image timestamps from file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    timestamps = []
    for line in lines:
        timestamp = float(line.strip().split()[0])  
        timestamps.append(timestamp)
    return np.array(timestamps)[:, np.newaxis].astype(np.float32)  

def load_images(data_dir):
    """Load images from file"""
    image_dir = os.path.join(data_dir, f"images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path) # BGR
        images.append(image)

    images = np.array(images)
    return images

def load_data(data_dir):
    calib_file = os.path.join(data_dir, "calib.txt")
    gt_poses_file = os.path.join(data_dir, "groundtruth.txt")
    imu_file = os.path.join(data_dir, "imu.txt")
    events_file = os.path.join(data_dir, "events.txt")
    image_timestamps_file = os.path.join(data_dir, "images.txt")

    calib = load_projection_matrix_for_camera(calib_file)
    gt_poses = load_2d_array(gt_poses_file)
    imu_data = load_2d_array(imu_file)
    events_data = load_2d_array(events_file)
    images = load_images(data_dir)
    image_timestamps = load_image_timestamps(image_timestamps_file)

    return {
        "calib": calib,
        "events_data": events_data,
        "gt_poses": gt_poses,
        "images": images,
        "image_timestamps": image_timestamps,
        "imu_data": imu_data,
    }

if __name__ == "__main__":
    data_name = "shapes_6dof"
    dataset_path = f"data/{data_name}"
    data = load_data(dataset_path)
    print("Shapes 6DOF data loaded successfully!")
    print("Calibration matrix shape:", data["calib"].shape)
    print("Ground truth poses shape:", data["gt_poses"].shape)
    print("IMU data shape:", data["imu_data"].shape)
    print("Events data shape:", data["events_data"].shape)
    print("Images shape:", data["images"].shape)
    print("Image timestamps shape:", data["image_timestamps"].shape)

    """
    Shapes 6DOF data loaded successfully!
    Calibration matrix shape: (3, 3)
    Ground truth poses shape: (11862, 8)
    IMU data shape: (59638, 7)
    Events data shape: (17962477, 4)
    Images shape: (1356, 180, 240, 3)
    Image timestamps shape: (1356, 1)
    """

