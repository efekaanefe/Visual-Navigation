import numpy as np
import cv2
import os


def calculate_trajectory_error(est_poses, gt_poses):
    error = est_poses - gt_poses
    error = np.linalg.norm(error, axis=1)
    error = np.linalg.norm(error, axis=0)
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
def load_projection_matrices(file_path):
    calib = {
        "left": load_projection_matrix_for_camera(file_path, is_left_camera=True),
        "right": load_projection_matrix_for_camera(file_path, is_left_camera=False)
    }
    return calib 

def load_projection_matrix_for_camera(file_path, is_left_camera = True):
    """Return the i-th calibration matrix (3x4)"""
    index = 0 if is_left_camera else 1
    with open(file_path, 'r') as f:
        lines = f.readlines()
    line = lines[index].strip()
    values = np.array(line.split(), dtype=np.float32)
    P = values.reshape(3, 4)  # projection matrix
    return P

def load_gt_poses(file_path):
    """Load poses from file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    poses = []
    for line in lines:
        pose = np.array(line.strip().split(), dtype=np.float32)
        pose = pose.reshape(3, 4)
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    
    return np.array(poses)

def load_images(data_dir, is_left_camera=True):
    """Load images from file"""
    appendix = "l" if is_left_camera else "r"
    image_dir = os.path.join(data_dir, f"image_{appendix}")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path) # BGR
        images.append(image)

    return images

if __name__ == "__main__":
    projection_matrices = load_projection_matrices("data/KITTI/2/calib.txt")
    poses = load_gt_poses("data/KITTI/2/poses.txt")
    images = load_images("data/KITTI/2", is_left_camera=True)
    print(projection_matrices["left"])
    print(projection_matrices["right"])
    print(poses[0])
    print(type(images[0]))
    print(images[0].shape)