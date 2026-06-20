import cv2
import numpy as np


def make_extrinsic(rotation, translation):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation.reshape(3)
    return extrinsic


def invert_extrinsic(rotation, translation):
    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ translation.reshape(3)
    return inverse_rotation, inverse_translation


def camera_center(rotation, translation):
    return -rotation.T @ translation.reshape(3)


def rotation_to_vector(rotation):
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return rotation_vector.reshape(3)


def vector_to_rotation(rotation_vector):
    rotation, _ = cv2.Rodrigues(rotation_vector.reshape(3, 1))
    return rotation
