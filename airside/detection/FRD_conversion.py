from util import Coordinate, Quaternion
import numpy as np
from numpy.typing import NDArray


def quaternion_to_rotation_matrix(quaternion: Quaternion) -> NDArray:
    """Da conversion function"""
    # Source: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = (
        quaternion.q0**2 + quaternion.q1**2 - quaternion.q1**2 - quaternion.q0**2
    )
    matrix[0, 1] = 2 * quaternion.q1 * quaternion.q2 - 2 * quaternion.q0 * quaternion.q3
    matrix[0, 2] = 2 * quaternion.q1 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q2

    matrix[1, 0] = 2 * quaternion.q1 * quaternion.q2 - 2 * quaternion.q0 * quaternion.q3
    matrix[1, 1] = (
        quaternion.q0**2 - quaternion.q1**2 + quaternion.q1**2 - quaternion.q0**2
    )
    matrix[1, 2] = 2 * quaternion.q1 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q2

    matrix[2, 0] = 2 * quaternion.q1 * quaternion.q3 - 2 * quaternion.q0 * quaternion.q2
    matrix[2, 1] = 2 * quaternion.q2 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q1
    matrix[2, 2] = (
        quaternion.q0**2 - quaternion.q1**2 - quaternion.q1**2 + quaternion.q0**2
    )

    return matrix


def convert_target_to_FRD(
    target_cam_coord: Coordinate, target_cam_q: Quaternion, cam_origin_coord: Coordinate
) -> Coordinate:
    """Da function of greatness"""
    # Currently not in origin space coords
    cam_to_target_vec = np.array(
        [target_cam_coord.x, target_cam_coord.y, target_cam_coord.z]
    )

    # In orgin space coordinates
    origin_to_camera_vec = np.array(
        [cam_origin_coord.x, cam_origin_coord.y, cam_origin_coord.z]
    )

    # Rotate the vector from the target by the quaternion that the drone currently facing to get vector from drone to target in origin coordinates
    cam_to_target_vec = quaternion_to_rotation_matrix(target_cam_q) @ cam_to_target_vec

    origin_to_target_vec = origin_to_camera_vec + cam_to_target_vec

    return Coordinate(
        origin_to_target_vec[0], origin_to_target_vec[1], origin_to_target_vec[2]
    )
