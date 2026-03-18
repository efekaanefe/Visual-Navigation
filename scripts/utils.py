import numpy as np
import cv2
import os


def calculate_trajectory_error(est_poses, gt_poses):
    error = est_poses - gt_poses
    error = np.linalg.norm(error, axis=1)
    return error


def get_intrinsic(projection_matrix):
    return projection_matrix[:3, :3]


def get_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def load_projection_matrix_for_camera(file_path):
    index = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()

    line = lines[index].strip()
    values = np.array(line.split(), dtype=np.float32)
    P = values.reshape(3, 3)
    return P


def load_2d_array(file_path, max_rows=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if max_rows is not None:
        lines = lines[:max_rows]

    arrays = []
    for line in lines:
        array = np.array(line.strip().split(), dtype=np.float32)
        arrays.append(array)

    return np.array(arrays)


def load_image_timestamps(file_path, max_rows=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if max_rows is not None:
        lines = lines[:max_rows]

    timestamps = []
    for line in lines:
        timestamp = float(line.strip().split()[0])
        timestamps.append(timestamp)

    return np.array(timestamps)[:, np.newaxis].astype(np.float32)


def load_images(data_dir, max_images=None):
    image_dir = os.path.join(data_dir, "images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    if max_images is not None:
        image_files = image_files[:max_images]

    image_paths = [os.path.join(image_dir, f) for f in image_files]

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)

    return np.array(images)


def load_data(data_dir, fraction=1.0):
    assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"

    calib_file = os.path.join(data_dir, "calib.txt")
    gt_poses_file = os.path.join(data_dir, "groundtruth.txt")
    imu_file = os.path.join(data_dir, "imu.txt")
    events_file = os.path.join(data_dir, "events.txt")
    image_timestamps_file = os.path.join(data_dir, "images.txt")

    with open(image_timestamps_file, 'r') as f:
        total_images = len(f.readlines())

    num_images = int(total_images * fraction)

    print(f"[INFO] Loading {num_images}/{total_images} images ({fraction*100:.1f}%)")

    image_timestamps = load_image_timestamps(image_timestamps_file, num_images)
    images = load_images(data_dir, num_images)
    gt_poses = load_2d_array(gt_poses_file, num_images)

    imu_total = sum(1 for _ in open(imu_file))
    imu_rows = int(imu_total * fraction)

    events_total = sum(1 for _ in open(events_file))
    events_rows = int(events_total * fraction)

    print(f"[INFO] IMU rows: {imu_rows}/{imu_total}")
    print(f"[INFO] Event rows: {events_rows}/{events_total}")

    imu_data = load_2d_array(imu_file, imu_rows)
    events_data = load_2d_array(events_file, events_rows)

    calib = load_projection_matrix_for_camera(calib_file)

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

    data = load_data(dataset_path, fraction=0.1)

    print("\n--- Loaded Data Shapes ---")
    print("Calibration matrix:", data["calib"].shape)
    print("Ground truth poses:", data["gt_poses"].shape)
    print("IMU data:", data["imu_data"].shape)
    print("Events data:", data["events_data"].shape)
    print("Images:", data["images"].shape)
    print("Image timestamps:", data["image_timestamps"].shape)
