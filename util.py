"""
Utility classes and constants for drone communication and data structures.

This module provides core data structures and enums used throughout the drone control system,
including local coordinate representation, MAVLink message types, and RC channel data.
"""

from enum import Enum

UINT16_MAX = 65535

# MAVLink communication constants
AIRSIDE_COMPONENT_ID = 191
MAVLINK_TCP_HOST = "127.0.0.1"
MAVLINK_TCP_PORT = 14550
MAVLINK_RECEIVE_TIMEOUT_SEC = 1


class MavlinkMessageType(Enum):
    """MAVLink message types used in drone communication"""

    GLOBAL_POSITION_INT = "GLOBAL_POSITION_INT"
    RC_CHANNELS = "RC_CHANNELS"


class Colour:
    def __init__(
        self,
        name: str,
        lower_hsv: tuple[int, int, int],
        upper_hsv: tuple[int, int, int],
    ):
        self.name = name
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv

    def __str__(self):
        return f"({self.name}, {self.lower_hsv}, {self.upper_hsv})"

    def __repr__(self):
        return f"Colour(name={self.name}, lower_hsv={self.lower_hsv}, upper_hsv={self.upper_hsv})"


class Colours(Enum):
    RED = Colour("Red", (0, 100, 100), (10, 255, 255))
    GREEN = Colour("Green", (36, 255, 255), (70, 255, 255))
    BLACK = Colour("Black", (0, 0, 0), (255, 255, 20))
    WHITE = Colour("White", (0, 0, 200), (255, 20, 255))
    BLUE = Colour("Blue", (100, 100, 100), (130, 255, 255))
    YELLOW = Colour("Yellow", (20, 100, 100), (30, 255, 255))


class Direction(Enum):
    """Cardinal directions"""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class Coordinate:
    """Represents a local coordinate with x, y, and z components."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Target:
    """Represents a target with a color and location."""

    def __init__(self, colour: Colour, location: Coordinate):
        self.colour = colour
        self.location = location

    def __str__(self):
        return f"(colour={self.colour}, location={self.location})"


class MappedTarget:
    """Represents a mapped target with a color, building-relative location, and cardinal direction."""

    def __init__(
        self,
        colour: Colours,
        location: Coordinate,
        direction: Direction,
        wall_target: bool = True,
    ):
        self.colour = colour
        self.location = location
        self.direction = direction
        self.wall_target = wall_target

    def __str__(self):
        return f"(colour={self.colour}, location={self.location}, cardinal_direction={self.direction}, wall_target={self.wall_target})"


class RCChannel:
    """Represents a single RC channel with raw value and activity status."""

    def __init__(self, channel: int, raw: int = 0, is_active: bool = False):
        self.channel = channel
        self.raw = raw
        self.is_active = is_active

    def __str__(self):
        return f"({self.channel}, {self.raw}, {self.is_active})"

    def __repr__(self):
        return f"RCChannel(channel={self.channel}, raw={self.raw}, is_active={self.is_active})"


class Vector3d:
    """Represents a 3D vector with x, y, and z components."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Plane:
    """Represents a plane in 3D space defined by an offset from origin and a normal vector."""

    def __init__(self, normal: Vector3d, offset: float):
        self.normal = normal
        self.offset = offset

    def __str__(self):
        return f"(offset={self.offset}, normal={self.normal})"


class Quaternion:
    """Represents a quaternion in 3D space with q0, q1, q2, q3 components"""

    def __init__(self, q0: float, q1: float, q2: float, q3: float):
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def __str__(self):
        return f"(q0: {self.q0}, q1: {self.q1}, q2: {self.q2}, q3: {self.q3})"
