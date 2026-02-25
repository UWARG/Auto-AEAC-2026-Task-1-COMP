import pytest
from util import Coordinate, Quaternion 
from airside.detection.FRD_conversion import convert_target_to_FRD

def test_zero_conversion():
    camera_to_target = Coordinate(0, 0, 0)
    drone_rotation = Quaternion(0, 1, 0, 0)
    origin_to_camera = Coordinate(0, 0, 0)

    result = convert_target_to_FRD(camera_to_target, drone_rotation, origin_to_camera)

    expected = Coordinate(0, 0, 0)
    print(f"Expected: {expected} | Result: {result}")

    assert expected.x == pytest.approx(result.x)
    assert expected.y == pytest.approx(result.y)
    assert expected.z == pytest.approx(result.z)

def test_2d_simple_conversion():
    camera_to_target = Coordinate(10, 0, 0)
    drone_rotation = Quaternion(1, 0, 0, 1)
    origin_to_camera = Coordinate(10, 0, 0)

    result = convert_target_to_FRD(camera_to_target, drone_rotation, origin_to_camera)

    expected = Coordinate(10, 10, 0)

    print(f"Expected: {expected} | Result: {result}")

    assert expected.x == pytest.approx(result.x)
    assert expected.y == pytest.approx(result.y)
    assert expected.z == pytest.approx(result.z)

def test_2d_complex_conversion():
    camera_to_target = Coordinate(10, 10, 10)
    drone_rotation = Quaternion(1, 1, 1, 1)
    origin_to_camera = Coordinate(10, 10, 10)

    result = convert_target_to_FRD(camera_to_target, drone_rotation, origin_to_camera)

    expected = Coordinate(20, 20, 20)

    print(f"Expected: {expected} | Result: {result}")

    assert expected.x == pytest.approx(result.x)
    assert expected.y == pytest.approx(result.y)
    assert expected.z == pytest.approx(result.z)

