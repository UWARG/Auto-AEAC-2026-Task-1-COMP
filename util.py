"""
Utility classes and constants for drone communication and data structures.

This module provides core data structures and enums used throughout the drone control system,
including local coordinate representation, MAVLink message types, and RC channel data.
"""

from enum import Enum

UINT16_MAX = 65535

# MAVLink communication constants
AIRSIDE_COMPONENT_ID = 191
MAVLINK_TCP_HOST = '127.0.0.1'
MAVLINK_TCP_PORT = 14550
MAVLINK_RECEIVE_TIMEOUT_SEC = 1


class MavlinkMessageType(Enum):
    """MAVLink message types used in drone communication"""
    
    GLOBAL_POSITION_INT = "GLOBAL_POSITION_INT"
    RC_CHANNELS = "RC_CHANNELS"


class Colour(Enum):
    """Target colors"""

    RED = "RED"
    GREEN = "GREEN"
    BLACK = "BLACK"
    BLUE = "BLUE"
    YELLOW = "YELLOW"


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

    def __init__(self, colour: Colour, location: Coordinate, direction: Direction):
        self.colour = colour
        self.location = location
        self.direction = direction

    def __str__(self):
        return f"(colour={self.colour}, location={self.location}, cardinal_direction={self.direction})"


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
