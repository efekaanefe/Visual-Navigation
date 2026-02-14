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



def estimate_depth(image_left, image_right, P_left, P_right):
    window_size = 10 
    min_disp = 0
    num_disp = 16 * 6
    
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = window_size,
                                   P1 = 8 * 3 * window_size**2,
                                   P2 = 32 * 3 * window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32)

    gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    f = P_left[0, 0]
    b = -P_right[0, 3] / P_right[0, 0]
    
    valid_mask = disparity > 0
    
    depth = np.zeros_like(disparity)
    
    depth[valid_mask] = (f * b) / disparity[valid_mask]
    
    depth[depth > 100] = 100 
    
    return depth


