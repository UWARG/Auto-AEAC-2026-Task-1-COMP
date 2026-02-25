from util import Coordinate, Quaternion
import numpy as np
from scipy import from_quat

def convert_target_to_FRD(
    cam_target_coord: Coordinate, origin_cam_q: Quaternion, origin_cam_coord: Coordinate
) -> Coordinate:
    """
    :cam_target_coord: The coordinates of the target from the drone's FRD coordinate space
    :origin_cam_q: The orientation of the drone from the origin's FRD coordinate space
    :origin_cam_coord: The coordinate of the drone from the origin's FRD coordinate space

    :return: The coordinate of the target from the origin's FRD coordinate space
    """
    # Currently not in origin space coords
    cam_to_target_vec = np.array(
        [cam_target_coord.x, cam_target_coord.y, cam_target_coord.z]
    )

    # In orgin space coordinates
    origin_to_camera_vec = np.array(
        [origin_cam_coord.x, origin_cam_coord.y, origin_cam_coord.z]
    )

    # Rotate the vector from the target by the quaternion that the drone currently facing to get vector from drone to target in origin coordinates
    cam_to_target_vec = np.asarray(from_quat(origin_cam_q)) @ cam_to_target_vec

    origin_to_target_vec = origin_to_camera_vec + cam_to_target_vec

    return Coordinate(
        origin_to_target_vec[0], origin_to_target_vec[1], origin_to_target_vec[2]
    )
