from abc import ABC, abstractmethod

import cv2


class FeatureExtractor(ABC):
    """Detects keypoints and computes descriptors for a grayscale image.

    Defined as an abstract base so the pipeline depends on the abstraction and a
    different detector (ORB, AKAZE, ...) can be substituted without touching the
    rest of the system.
    """

    @abstractmethod
    def detect_and_compute(self, gray_image):
        """Return ``(keypoints, descriptors)`` for ``gray_image``."""


class SiftFeatureExtractor(FeatureExtractor):
    def __init__(self, max_features):
        self._detector = cv2.SIFT_create(nfeatures=max_features)

    def detect_and_compute(self, gray_image):
        return self._detector.detectAndCompute(gray_image, None)
