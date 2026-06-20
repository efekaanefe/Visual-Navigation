import cv2
import numpy as np


class FeatureMatcher:
    """Nearest-neighbour descriptor matching with Lowe's ratio test."""

    def __init__(self, lowe_ratio):
        self._lowe_ratio = lowe_ratio
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, query_descriptors, train_descriptors):
        """Return matched index pairs as an ``(M, 2)`` array ``[query_idx, train_idx]``."""
        if not self._has_enough_descriptors(query_descriptors, train_descriptors):
            return np.empty((0, 2), dtype=np.int64)

        knn_matches = self._matcher.knnMatch(query_descriptors, train_descriptors, k=2)
        index_pairs = []
        for candidate_pair in knn_matches:
            if len(candidate_pair) < 2:
                continue
            nearest, second_nearest = candidate_pair
            if nearest.distance < self._lowe_ratio * second_nearest.distance:
                index_pairs.append((nearest.queryIdx, nearest.trainIdx))
        return np.array(index_pairs, dtype=np.int64).reshape(-1, 2)

    @staticmethod
    def _has_enough_descriptors(query_descriptors, train_descriptors):
        if query_descriptors is None or train_descriptors is None:
            return False
        return len(query_descriptors) >= 2 and len(train_descriptors) >= 2
