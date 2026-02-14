from scripts.utils import (
    load_projection_matrix_for_camera, 
    load_gt_poses, 
    load_images, 
    get_intrinsic, 
    get_transformation_matrix
)

import os
import cv2

import numpy as np



class VisualOdometry:
    def __init__(self, data_dir = "data/KITTI/1", is_left_camera=True, n_features = 1000):
        # Load Data
        self.P = load_projection_matrix_for_camera(os.path.join(data_dir, "calib.txt"), is_left_camera=is_left_camera)
        self.K = get_intrinsic(self.P)
        self.gt_poses = load_gt_poses(os.path.join(data_dir, "poses.txt"))
        self.images = load_images(data_dir, is_left_camera=is_left_camera)

        # Feature Detector
        self.n_features = n_features
        self.detector = cv2.SIFT_create(nfeatures=self.n_features) 
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def get_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
        
    def match_keypoints(self, descriptors1, descriptors2):
        matches = self.matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def estimate_pose(self, keypoints1, keypoints2, matches):
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)

        T = get_transformation_matrix(R, t)
        return T

    def estimate_trajectory(self):
        current_pose = np.eye(4)
        poses = [current_pose]
        for i in range(len(self.images) - 1):
            keypoints1, descriptors1 = self.get_keypoints(self.images[i])
            keypoints2, descriptors2 = self.get_keypoints(self.images[i+1])
            matches = self.match_keypoints(descriptors1, descriptors2)
            T = self.estimate_pose(keypoints1, keypoints2, matches)
            current_pose = current_pose @ T
            poses.append(current_pose)
        return np.array(poses)