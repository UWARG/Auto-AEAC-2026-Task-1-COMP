"""
Utility classes and constants for drone communication and data structures.

This module provides core data structures and enums used throughout the drone control system,
including local coordinate representation, MAVLink message types, and RC channel data.
"""

from enum import Enum
import cv2
import numpy as np
import abc

import libcamera
import picamera2

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

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False


class Target:
    """Represents a target with a color and location."""

    def __init__(self, colour: Colours, location: Coordinate):
        self.colour = colour
        self.location = location

    def __str__(self):
        return f"{self.colour.name}, {self.location}"


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


class ImageLocation:
    """Represents point on an image (+x right, +y down)"""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"


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
    """Represents a quaternion in 3D space with w, x, y, z components"""

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"(w: {self.w}, x: {self.x}, y: {self.y}, z: {self.z})"

    def to_array(self) -> list[float]:
        return [self.w, self.x, self.y, self.z]

    def __eq__(self, other):
        if (
            self.w == other.w
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        ):
            return True
        else:
            return False


class BaseCameraDevice(abc.ABC):
    """
    Abstract class for camera device implementations.
    """

    @classmethod
    @abc.abstractmethod
    def create(
        cls,
        width: int,
        height: int,
        config: object,
    ) -> "tuple[True, BaseCameraDevice] | tuple[False, None]":
        """
        Abstract create method.

        width: Width of the camera.
        height: Height of the camera.
        config: Configuration.

        Return: Success, camera object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, class_private_create_key: object, camera: object) -> None:
        """
        Abstract private constructor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __del__(self) -> None:
        """
        Destructor. Release hardware resources.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> tuple[True, np.ndarray] | tuple[False, None]:
        """
        Takes a picture with camera device.

        Return: Success, image with shape (height, width, channels in BGR).
        """
        raise NotImplementedError


class ConfigOpenCV:
    """
    Configuration for the OpenCV camera.

    Attributes
    ----------
    device_index : int
        Index of the camera device.
    """

    def __init__(self, device_index: int) -> None:
        """
        Initialize OpenCV camera configuration.

        Parameters
        ----------
        device_index : int
            Index of the camera device.
        """
        self.device_index = device_index


class CameraOption(enum.Enum):
    """
    enum for type of camera object to create.
    """

    OPENCV = 0
    PICAM2 = 1
    ARDUCAMIR = 2


def create_camera(
    camera_option: CameraOption,
    width: int,
    height: int,
    config: ConfigOpenCV | None,
) -> tuple[True, BaseCameraDevice] | tuple[False, None]:
    """
    Create a camera object based off of given parameters.

    Return: Success, camera device object.
    """
    match camera_option:
        case CameraOption.OPENCV:
            return CameraOpenCV.create(width, height, config)

    return False, None


class CameraOpenCV(BaseCameraDevice):
    """
    Class for the OpenCV implementation of the camera.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, width: int, height: int, config: ConfigOpenCV
    ) -> "tuple[True, CameraOpenCV] | tuple[False, None]":
        """
        OpenCV Camera.

        width: Width of the camera.
        height: Height of the camera.
        config: Configuration of for OpenCV camera.

        Return: Success, camera object.
        """
        if width <= 0:
            return False, None

        if height <= 0:
            return False, None

        camera = cv2.VideoCapture(config.device_index)
        if not camera.isOpened():
            return False, None

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        set_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        set_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if set_width != width or set_height != height:
            return False, None

        return True, CameraOpenCV(cls.__create_key, camera)

    def __init__(
        self, class_private_create_key: object, camera: cv2.VideoCapture
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert (
            class_private_create_key is CameraOpenCV.__create_key
        ), "Use create() method."

        self.__camera = camera

    def __del__(self) -> None:
        """
        Destructor. Release hardware resources.
        """
        self.__camera.release()

    def run(self) -> tuple[True, np.ndarray] | tuple[False, None]:
        """
        Takes a picture with OpenCV camera.

        Return: Success, image with shape (height, width, channels in BGR).
        """
        result, image_data = self.__camera.read()
        if not result:
            return False, None

        return True, image_data


class ConfigPiCamera2:
    """
    Configuration for the PiCamera.
    """

    def __init__(
        self,
        timeout: float = 1.0,
        exposure_time: int = 250,
        analogue_gain: float = 64.0,
        contrast: float = 1.0,
        maybe_lens_position: float | None = None,
    ) -> None:
        """
        timeout: Getting image timeout in seconds.

        exposure_time: Microseconds.
        analogue_gain: 0.0 to 64.0 . ISO = Analogue gain * Digital gain * 100 .
        contrast: 0.0 to 32.0 . 0.0 is no contrast, 1.0 is normal contrast, higher is more contrast.
        lens_position: Position of the lens is dioptres (reciprocal of metres: 1/m ) (0 means infinite distance).
        """
        self.timeout = timeout

        self.exposure_time = exposure_time
        self.analogue_gain = analogue_gain
        self.contrast = contrast
        self.maybe_lens_position = maybe_lens_position

    def to_dict(self) -> dict[str, int | float]:
        """
        Dictionary containing camera controls.
        """
        camera_controls = {
            "ExposureTime": self.exposure_time,
            "AnalogueGain": self.analogue_gain,
            "Contrast": self.contrast,
        }

        if self.maybe_lens_position is not None:
            camera_controls["LensPosition"] = self.maybe_lens_position
            camera_controls["AfMode"] = libcamera.controls.AfModeEnum.Manual
        else:
            camera_controls["LensPosition"] = 0.0
            camera_controls["AfMode"] = libcamera.controls.AfModeEnum.Auto

        return camera_controls


class CameraPiCamera2(BaseCameraDevice):
    """
    Class for the Picamera2 implementation of the camera.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, width: int, height: int, config: ConfigPiCamera2
    ) -> "tuple[True, CameraPiCamera2] | tuple[False, None]":
        """
        Picamera2 Camera.

        width: Width of the camera.
        height: Height of the camera.
        config: Configuration for PiCamera2 camera.

        Return: Success, camera object.
        """

        if width <= 0:
            return False, None

        if height <= 0:
            return False, None

        try:
            camera = picamera2.Picamera2()

            camera_config = camera.create_preview_configuration(
                {"size": (width, height), "format": "RGB888"}
            )
            camera.configure(camera_config)
            camera.start()
            controls = config.to_dict()
            camera.set_controls(controls)

            return True, CameraPiCamera2(cls.__create_key, camera, config)
        except RuntimeError:
            return False, None

    def __init__(
        self,
        class_private_create_key: object,
        camera: picamera2.Picamera2,  # type: ignore
        config: ConfigPiCamera2,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert (
            class_private_create_key is CameraPiCamera2.__create_key
        ), "Use create() method."

        self.__camera = camera
        self.__config = config

    def __del__(self) -> None:
        """
        Destructor. Release hardware resources.
        """
        self.__camera.close()

    def run(self) -> tuple[True, np.ndarray] | tuple[False, None]:
        """
        Takes a picture with Picamera2 camera.

        Return: Success, image with shape (height, width, channels in BGR).
        """
        try:
            image_data = self.__camera.capture_array(wait=self.__config.timeout)
        except TimeoutError:
            return False, None

        return True, image_data
