import numpy as np
import cv2

class EventClassicalVO:
    def __init__(self, intrinsic_matrix, delta_t_sec=0.05, n_features=1000):
        self.K = intrinsic_matrix
        self.delta_t = delta_t_sec
        
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def accumulate_frame(self, events, image_shape):
        """Maps continuous spatial-temporal events to a discrete 2D intensity gradient."""
        H, W = image_shape
        frame = np.zeros((H, W), dtype=np.uint16)
        
        x = np.asarray(events[:, 1], dtype=np.int32)
        y = np.asarray(events[:, 2], dtype=np.int32)
        
        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        np.add.at(frame, (y[valid_mask], x[valid_mask]), 50)
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def match_features(self, des1, des2):
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:max(8, int(len(matches) * 0.3))]

    def estimate_pose(self, kp1, kp2, matches):
        if len(matches) < 8:
            return np.eye(4), False

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return np.eye(4), False

        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T, True

    def estimate_trajectory(self, events, image_shape):
        """Integrates relative transformations T_{k-1, k} into global pose T_W."""
        t_start = events[0, 0]
        t_end = events[-1, 0]
        current_t = t_start
        
        current_pose = np.eye(4)
        poses = [current_pose]
        timestamps = [current_t]
        
        prev_kps, prev_descs = None, None

        while current_t < t_end:
            window_mask = (events[:, 0] >= current_t) & (events[:, 0] < (current_t + self.delta_t))
            event_batch = events[window_mask]
            
            if len(event_batch) > 0:
                frame = self.accumulate_frame(event_batch, image_shape)
                kps, descs = self.detector.detectAndCompute(frame, None)
                
                if prev_descs is not None and descs is not None:
                    matches = self.match_features(prev_descs, descs)
                    T, success = self.estimate_pose(prev_kps, kps, matches)
                    
                    if success:
                        current_pose = current_pose @ T
                
                # Update state regardless of success to re-initialize tracking
                prev_kps, prev_descs = kps, descs
            
            poses.append(current_pose)
            timestamps.append(current_t)
            current_t += self.delta_t
            
        return np.array(poses), np.array(timestamps)
