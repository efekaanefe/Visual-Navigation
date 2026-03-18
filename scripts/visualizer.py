import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def poses_to_SE3(gt_poses):
    """
    Convert poses in shape (N, 8) [t, tx, ty, tz, qx, qy, qz, qw] 
    to SE(3) matrices of shape (N, 4, 4)
    """
    T = np.zeros((gt_poses.shape[0], 4, 4))
    for i, p in enumerate(gt_poses):
        t = p[1:4]
        q = p[4:8]  # qx, qy, qz, qw
        rot = R.from_quat(q).as_matrix()
        T[i, :3, :3] = rot
        T[i, :3, 3] = t
        T[i, 3, 3] = 1.0
    return T


class Visualizer:
    def __init__(self):
        pass

    def plot_trajectory(self, poses):
        x = poses[:, 0, 3]
        y = poses[:, 1, 3]
        z = poses[:, 2, 3]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z)

        ax.set_title("3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Equal scaling (important, otherwise trajectory looks distorted)
        max_range = max(
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ) / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def plot_imu(self, imu_data):
        acc = imu_data[:, 1:4]
        gyro = imu_data[:, 4:7]

        fig = plt.figure(figsize=(12, 6))

        # ---- Accelerometer 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(acc[:, 0], acc[:, 1], acc[:, 2])
        ax1.set_title("Accelerometer (3D)")
        ax1.set_xlabel("ax")
        ax1.set_ylabel("ay")
        ax1.set_zlabel("az")

        # ---- Gyroscope 3D plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(gyro[:, 0], gyro[:, 1], gyro[:, 2])
        ax2.set_title("Gyroscope (3D)")
        ax2.set_xlabel("gx")
        ax2.set_ylabel("gy")
        ax2.set_zlabel("gz")

        plt.tight_layout()
        plt.show()

    def visualize_images_stream(self, images, delay=30):
        while True:  
            for img in images:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Sequence", img_bgr)

                key = cv2.waitKey(max(1, delay)) & 0xFF
                if key == 27 or key == ord('q'):
                    cv2.destroyAllWindows()
                    return


    def visualize_events_stream(self, events, H=180, W=240, dt=0.01, speed=1.0):
        t0 = events[0, 0]
        t_end = events[-1, 0]

        while True:  
            current_t = t0  

            while current_t < t_end:
                mask = (events[:, 0] >= current_t) & (events[:, 0] < current_t + dt)
                batch = events[mask]

                canvas = np.zeros((H, W, 3), dtype=np.uint8)

                for e in batch:
                    _, x, y, p = e
                    x, y = int(x), int(y)

                    if 0 <= x < W and 0 <= y < H:
                        if p > 0:
                            canvas[y, x] = [255, 0, 0]
                        else:
                            canvas[y, x] = [0, 0, 255]

                cv2.imshow("Event Stream", canvas)

                delay = int(dt * 1000 / speed)
                key = cv2.waitKey(max(1, delay)) & 0xFF
                if key == 27 or key == ord('q'):
                    cv2.destroyAllWindows()
                    return

                current_t += dt

