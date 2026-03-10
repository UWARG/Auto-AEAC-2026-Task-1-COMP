import numpy as np
import math
from util import Plane, Target, MappedTarget, Direction, Coordinate, Colours, Vector3d


def locate_targets(
    planes: list[Plane], targets: list[Target], first_direction: Direction
) -> list[MappedTarget]:

    MARGIN = 0.5  # Margin of error in meteres where anything below this value can be considered negligeble
    ANGLE_THRESHOLD = 0.087  # ~= 5 degrees

    bottom_corners = []
    walls = {
        "close_x_wall": None,  # wall with cardinal direction, as first index in ordered dict
        "negative_y_wall": None,
        "far_x_wall": None,
        "positive_y_wall": None,
    }
    ground_vector = None
    ground_offset = None
    direction = ["NORTH", "EAST", "SOUTH", "WEST"]
    for plane in planes:
        plane_normal = np.array([plane.normal.x, plane.normal.y, plane.normal.z])
        if abs(plane_normal[2]) > MARGIN:  # if normal has some z component
            ground_vector = plane_normal
            ground_offset = plane.offset
            continue
        dp = np.dot(plane_normal, np.array([1, 0, 0]))
        angle = math.acos(np.clip(dp / (np.linalg.norm(plane_normal)), -1, 1))
        if angle < ANGLE_THRESHOLD:
            if not walls["close_x_wall"]:
                walls["close_x_wall"] = plane
            else:
                if abs(
                    walls["close_x_wall"].offset / walls["close_x_wall"].normal.x
                ) > abs(plane.offset):
                    walls["far_x_wall"] = walls["close_x_wall"]
                    walls["close_x_wall"] = plane
                else:
                    walls["far_x_wall"] = plane
        # Want perp1 to be the wall in the positive y direction
        else:
            if not walls["negative_y_wall"]:
                walls["negative_y_wall"] = plane
            else:
                if (
                    plane.normal.y > 0
                ):  # we only have to divide by normal.y for perp walls since their normals are opposite direction
                    walls["positive_y_wall"] = plane
                else:
                    walls["positive_y_wall"] = walls["negative_y_wall"]
                    walls["negative_y_wall"] = plane

    # TODO: TEMPORARY DEFAULT if any walls are missing
    if walls["close_x_wall"] is None:
        walls["close_x_wall"] = Plane(
            normal=Vector3d(1, 0, 0),
            offset=3.0
        )
    if walls["far_x_wall"] is None:
        walls["far_x_wall"] = Plane(
            normal=Vector3d(1, 0, 0),
            offset=6.0
        )
    if walls["negative_y_wall"] is None:
        walls["negative_y_wall"] = Plane(
            normal=Vector3d(0, -1, 0),
            offset=3.0
        )
    if walls["positive_y_wall"] is None:
        walls["positive_y_wall"] = Plane(
            normal=Vector3d(0, 1, 0),
            offset=3.0
        )
    if ground_vector is None:
        ground_vector = np.array([0, 0, 1])
        ground_offset = 0.0

    close_wall = np.array(
        [
            walls["close_x_wall"].normal.x,
            walls["close_x_wall"].normal.y,
            walls["close_x_wall"].normal.z,
        ]
    )
    far_wall = np.array(
        [
            walls[
                "far_x_wall"
            ].normal.x,  # far wall parallel to first wall with cardinal direction
            walls["far_x_wall"].normal.y,
            walls["far_x_wall"].normal.z,
        ]
    )
    for wall in [walls["negative_y_wall"], walls["positive_y_wall"]]:
        wall2 = np.array([wall.normal.x, wall.normal.y, wall.normal.z])
        matrix = np.vstack((close_wall, wall2, ground_vector))
        matrix2 = np.vstack((far_wall, wall2, ground_vector))
        b = np.array(
            [
                walls["close_x_wall"].offset,
                wall.offset,
                ground_offset,
            ]
        )
        b2 = np.array(
            [
                walls["far_x_wall"].offset,
                wall.offset,
                ground_offset,
            ]
        )
        corner = np.linalg.solve(matrix, b)
        corner2 = np.linalg.solve(matrix2, b2)
        # order corners to go clockwise as index increases
        if wall.normal.y < 0:
            bottom_corners.append(corner)  # front corner
            bottom_corners.append(corner2)  # back corner
        else:
            bottom_corners.append(corner2)  # back corner
            bottom_corners.append(corner)  # front corner

    target_locations = []
    for target in targets:
        right, up = 0, 0
        target_on_wall = False
        target_vector = np.array(
            [target.location.x, target.location.y, target.location.z]
        )
        for corner, wall in zip(bottom_corners, list(walls.values())):
            vector = np.subtract(target_vector, corner)
            normal_vector = np.array([wall.normal.x, wall.normal.y, wall.normal.z])
            dp = np.dot(vector, normal_vector)
            angle = math.acos(
                np.clip(
                    dp / (np.linalg.norm(vector) * np.linalg.norm(normal_vector)), -1, 1
                )
            )

            if abs(math.pi / 2 - angle) < ANGLE_THRESHOLD:
                if abs(vector[0]) < MARGIN or abs(vector[0]) < abs(vector[1]):

                    right, up = abs(vector[1]), abs(vector[2])
                    relative_position = Coordinate(right, up, 0)
                else:
                    right, up = abs(vector[0]), abs(vector[2])
                    relative_position = Coordinate(right, up, 0)
                if abs(vector[2]) < min(
                    abs(vector[0]), abs(vector[1])
                ):  # more likely to be on the ground so continue
                    continue
                target_on_wall = True
                index = direction.index(first_direction.value)
                index2 = list(walls.values()).index(wall)
                wall_direction = direction[(index + index2) % 4]
                target_locations.append(
                    MappedTarget(
                        Colours[target.colour.name.upper()],
                        relative_position,
                        Direction[wall_direction],
                        True,
                    )
                )
                break

            # value from right of bottom left corner will always be in +x or +y dir. Other component direction
            # should be 0 for "vector"
        # want to find corner regions
        position = (0, 0)
        if not target_on_wall:
            wall = None
            if (
                target.location.y
                < walls["negative_y_wall"].offset / walls["negative_y_wall"].normal.y
            ):
                if (
                    target.location.x
                    > walls["far_x_wall"].offset / walls["far_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[2])
                    position = (abs(vector[1]), abs(vector[0]))
                    wall = walls["far_x_wall"]
                elif (
                    target.location.x
                    < walls["close_x_wall"].offset / walls["close_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[1])
                    position = (abs(vector[0]), abs(vector[1]))
                    wall = walls["negative_y_wall"]
                else:
                    vector = np.subtract(target_vector, bottom_corners[1])
                    position = (abs(vector[0]), abs(vector[1]))
                    wall = walls["negative_y_wall"]

            elif (
                target.location.y
                > walls["positive_y_wall"].offset / walls["positive_y_wall"].normal.y
            ):
                if (
                    target.location.x
                    > walls["far_x_wall"].offset / walls["far_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[3])
                    position = (abs(vector[0]), abs(vector[1]))
                    wall = walls["positive_y_wall"]
                elif (
                    target.location.x
                    < walls["close_x_wall"].offset / walls["close_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[0])
                    position = (abs(vector[1]), abs(vector[0]))
                    wall = walls["close_x_wall"]
                else:
                    vector = np.subtract(target_vector, bottom_corners[3])
                    position = (abs(vector[0]), abs(vector[1]))
                    wall = walls["positive_y_wall"]
            # for in between two perpendicular walls
            else:
                if (
                    target.location.x
                    > walls["far_x_wall"].offset / walls["far_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[2])
                    wall = walls["far_x_wall"]
                    position = (abs(vector[1]), abs(vector[0]))
                elif (
                    target.location.x
                    < walls["close_x_wall"].offset / walls["close_x_wall"].normal.x
                ):
                    vector = np.subtract(target_vector, bottom_corners[0])
                    wall = walls["close_x_wall"]
                    position = (abs(vector[1]), abs(vector[0]))
                # for slight imperfections within margin of error
                elif (
                    abs(
                        target.location.x
                        - walls["close_x_wall"].offset / walls["close_x_wall"].normal.x
                    )
                    < MARGIN
                    or abs(
                        target.location.x
                        - walls["far_x_wall"].offset / walls["far_x_wall"].normal.x
                    )
                    < MARGIN
                ):
                    vector = np.subtract(target_vector, bottom_corners[0])
                    vector2 = np.subtract(target_vector, bottom_corners[2])
                    if np.linalg.norm(vector) < np.linalg.norm(vector2):
                        # change to position = (abs(vector[1]), -abs(vector[0])) if you want a negative value for position's forward value
                        position = (abs(vector[1]), 0)
                        wall = walls["close_x_wall"]
                    else:
                        # change to position = (abs(vector2[1]), -abs(vector2[0]))if you want a negative value for position's forward value
                        position = (abs(vector2[1]), 0)
                        wall = walls["far_x_wall"]

                else:  # otherwise invalid
                    raise Exception(f"Invalid target location {target.location}")

            index = direction.index(first_direction.value)
            index2 = list(walls.values()).index(wall)
            wall_direction = direction[(index + index2) % 4]
            relative_position = Coordinate(*position, 0)

            target_locations.append(
                MappedTarget(
                    Colours[target.colour.name.upper()],
                    relative_position,
                    Direction[wall_direction],
                    False,
                )
            )

    return target_locations
