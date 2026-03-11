import math
import pytest
from airside.detection.target_localization import get_target_coordinates
from util import ImageLocation, Coordinate

# @pytest.fixture
def test_localization():
    """
    Takes in list of test points and compares expected positions to calculated target position
    """

    detections = [ImageLocation(1124, 283), ImageLocation(1124, 311), ImageLocation(1052, 122)]
    z_dist = 15  # m
    roll = 0
    pitch = 0
    yaw = 0

    # Perform transformations on points
    calculated_points = get_target_coordinates(
        detections_=detections, roll=roll, pitch=pitch, yaw=yaw, dist_reading=z_dist
    )

    expected_positions = [
        (1.225, 0, 15),
        (1.109, 0, 15),
        (0.80625 + 1.225, -0.355, 15),
    ]  # m

    # Print results
    for i in range(len(calculated_points)):
        error_x = expected_positions[i][0] - calculated_points[i].x
        error_y = expected_positions[i][1] - calculated_points[i].y
        error_z = expected_positions[i][2] - calculated_points[i].z

        print(
            f"""Point {i+1}:\n
            Expected x: {expected_positions[i][0]:2f}m  Actual:{calculated_points[i].x:2f}m  Error:{error_x:2f}m \n
            Expected y: {expected_positions[i][1]:2f}m  Actual:{calculated_points[i].y:2f}m  Error:{error_y:2f}m \n
            Expected z: {expected_positions[i][2]:2f}m  Actual:{calculated_points[i].z:2f}m  Error:{error_z:2f}m \n\n
            """
        )

if __name__ == "__main__":
    test_localization()
