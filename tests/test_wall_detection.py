import pytest
from airside.post_processing.wall_detection import find_walls
import os


def test_wall_detection():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file = os.path.join(current_dir, "stereo_vision_data.ply")

    result = find_walls(file)

    expected_walls = 2
    result_walls = len(result)

    print(f"Expected: {expected_walls} walls | Result: {result_walls}")

    assert expected_walls == pytest.approx(result_walls)
