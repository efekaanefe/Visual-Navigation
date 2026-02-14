import cv2
import matplotlib.pyplot as plt
import numpy as np



class Visualizer:
    def __init__(self):
        pass

    def plot_trajectory(self, poses):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111) #
        
        # X-Z Plane (Top View for KITTI: X=right, Z=forward)
        ax.plot(poses[:, 0, 3], poses[:, 2, 3])
        ax.set_title("Top View")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.show()

    def plot_trajectory_comparison(self, est_poses, gt_poses):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Plot trajectories with markers
        ax.plot(est_poses[:, 0, 3], est_poses[:, 2, 3], label="Estimated", marker='.', markersize=2)
        ax.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], label="Ground Truth", marker='.', markersize=2)

        # Draw dashed lines connecting corresponding poses
        for i in range(len(est_poses)):
            ax.plot([est_poses[i, 0, 3], gt_poses[i, 0, 3]], 
                    [est_poses[i, 2, 3], gt_poses[i, 2, 3]], 
                    'k--', linewidth=0.5, alpha=0.5)

        ax.set_title("Top View")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True)
        ax.legend()
        ax.axis('equal')
        
        plt.tight_layout()
        plt.show()

    def plot_trajectory_error(self, error):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(error)
        ax.set_title("Trajectory Error")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Error")
        ax.legend()
        plt.tight_layout()
        plt.show()

    

    def visualize_keypoints(self, image, keypoints):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111) #
        ax.imshow(image)
        ax.scatter([kp.pt[0] for kp in keypoints], [kp.pt[1] for kp in keypoints], c='r', s=10)
        plt.tight_layout()
        plt.show()

    def visualize_matches(self, image1, image2, keypoints1, keypoints2, matches):
        img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB) 
        
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches)
        plt.title(f"Matches: {len(matches)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    def visualize_depth_map(self, depth_map):
        plt.figure(figsize=(12, 5))
        plt.imshow(depth_map, cmap='plasma')
        plt.title("Depth Map")
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.show()