import pytest
from airside.post_processing.wall_detection import find_walls
from airside.mavlink_comm import MavlinkComm
import logging

def test_wall_detection():
    
    main_logger = logging.getLogger("main")
    file = "stereo_vision_data.ply"

    result = find_walls(file, MavlinkComm(main_logger))

    expected_walls = 2
    result_walls = len(result)

    print(f"Expected: {expected_walls} walls | Result: {result_walls}")

    assert expected_walls == pytest.approx(result_walls)
