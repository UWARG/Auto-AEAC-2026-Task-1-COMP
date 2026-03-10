import math
import numpy as np
from util import Coordinate, ImageLocation, Vector3d

CAMERA_RESOLUTION = (2304.0, 1296.0)
CAMERA_RESOLUTION_CALIB = (640.0, 480.0)

X_SCALE = CAMERA_RESOLUTION[0] / CAMERA_RESOLUTION_CALIB[0]
Y_SCALE = CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION_CALIB[1]

# matrix for camera marked 'a'
# IMPORTANT: Should recalibrate for actual camera before use
CAM_MATRIX = np.array(
    [
        [1.28776791e3 * X_SCALE, 0, 3.0470598e2 * X_SCALE],
        [0, 1.293471159e3 * Y_SCALE, 2.0603105e2 * Y_SCALE],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def get_target_coordinates(
    detections_: list[ImageLocation],
    roll: float,
    pitch: float,
    yaw: float,
    dist_reading: float,
    roll_threshold: float,
    origin_translation: Vector3d,
) -> list[Coordinate]:
    """
    Calculates target positions in FRD coordinates using downwards camera detections,
    range finder reading and attitude
    Parameters:
        detections_: List of bounding boxes from target detections
        roll, pitch and yaw: euler angles of drone (degrees)
        dist_reading: range finder reading (meters)
        roll_threshold: maximum roll of the drone where detections are still accepted (degrees)
        origin_translation: vector represeting local position of the drone (m)
    Returns:
        - List of Coordinate objects corresponding to local positions of detections (FRD)
    """
    # Check conditions to run script
    if abs(roll) > roll_threshold:  # roll is beyond threshold
        return []

    if not detections_:  # no detections
        return []

    # === prep transformation matricies ===
    # cam to body
    R_cam_to_body = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)

    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    height = dist_reading * math.cos(abs(roll_rad))

    # body to frd
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    R_body_to_frd = np.array(
        [
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
        ],
        dtype=np.float32,
    )

    output = []

    for kp in detections_:

        # normalized camera direction matrix
        Q_cam = (
            np.linalg.inv(CAM_MATRIX)
            @ np.array([float(kp.x), float(kp.y), 1], dtype=np.float32).T
        )

        # rotate camera --> body --> frd
        Q_frd = R_body_to_frd @ R_cam_to_body @ Q_cam.T

        # scale based on altitude reading
        P_frd = (Q_frd * height) / Q_frd[2]

        # calculate local position of target
        local_coord_frd = Coordinate(
            P_frd[0] + origin_translation.x,
            P_frd[1] + origin_translation.y,
            P_frd[2] + origin_translation.z,
        )
        # add point to output
        output.append(local_coord_frd)

    return output
