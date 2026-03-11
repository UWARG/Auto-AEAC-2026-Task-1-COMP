import threading
import logging
import time
from typing import List

from .abstract_camera import AbstractCamera

from util import (
    Colours,
    Coordinate,
    Target,
    CameraOption,
    create_camera,
    ConfigPiCamera2,
)

import cv2
import numpy as np


class TargetDetection:
    colour: Colours
    x: float
    y: float
    circularity: float
    coverage: float
    area: float
    contour: np.ndarray = None  # OpenCV contour for visualization


class Arducam(AbstractCamera):
    def __init__(
        self,
        main_logger: logging.Logger,
        detections_logger: logging.Logger,
        detailed_detections_logger: logging.Logger,
        stop_event: threading.Event,
    ):

        super().__init__(
            main_logger, detections_logger, detailed_detections_logger, stop_event
        )

        while not self._initialize_camera():
            main_logger.error(f"Failed to initialize camera, retrying in 1 second...")
            time.sleep(1)

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1)

            frame = self.capture_frame()
            if frame is not None:
                targets = self.find_targets(frame)
                # best_target = self.get_best_target(targets)

                for i in targets:
                    target = Target(
                        color=i.colour,
                        location=Coordinate(x=i.x, y=i.y, z=0),
                    )

                self.main_logger.info(f"Detected target: {target}")

        self.main_logger.info("Stopping Arducam thread.")

    def _initialize_camera(self) -> bool:
        """
        Initialize camera hardware and apply settings.
        """
        # Initialize Raspberry Pi Camera Module 2 with specified camera index

        self.main_logger.info("Attempting to initialize rpi camera")

        config = ConfigPiCamera2(
            exposure_time=self.exposure_time, analogue_gain=self.analogue_gain
        )

        status, obj = create_camera(CameraOption.PICAM2, 500, 500, config)
        self._camera = obj
        if status:
            self.main_logger.info(
                f"picam2 camera {self.camera_index} initialized successfully"
            )
        else:
            self.main_logger.error(
                f"picam2 camera {self.camera_index} failed to initialize"
            )
        return status

    def capture_frame(self) -> np.ndarray | None:
        """
        Capture a single frame from the camera.
        """
        # OAK-D uses its own pipeline, not BaseCameraDevice

        if self._camera is None:
            self.main_logger.warning("Camera is None during frame capture.")
            return None
        try:
            frame = self._camera.capture_array()
            return frame
        except RuntimeError as e:
            self.main_logger.warning("Failed to capture frame from rpi camera")
            return None
        except Exception as e:
            self.main_logger.error(f"Exception during frame capture: {e}")
            return None

    def __enter__(self) -> "Camera":
        """Context manager entry - allows use with 'with' statement."""
        return self

    def find_targets(self, frame: np.ndarray) -> List[TargetDetection]:
        """
        Detect colored circular targets in the frame.
        Returns the color of the circular target closest to the frame center.
        """
        if frame is None or frame.size == 0:
            self.main_logger.warning("Invalid frame provided to colour_in_frame method")
            return []

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        targets = []

        for colour in [c.value for c in Colours]:
            mask = cv2.inRange(frame_hsv, colour.lower_hsv, colour.upper_hsv)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                # Calculate circularity: 4*pi*area / perimeter^2
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Check fill ratio: ensure the contour is solid, not hollow
                # Create a mask for just this contour
                contour_mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                # Count colored pixels within the contour
                colored_pixels_in_contour = cv2.countNonZero(
                    cv2.bitwise_and(mask, contour_mask)
                )
                fill_ratio = colored_pixels_in_contour / area if area > 0 else 0

                # Calculate center of this circular contour

                bx, by, bw, bh = cv2.boundingRect(contour)

                center_x = bx + (bw / 2.0)
                center_y = by + (bh / 2.0)

                # moments = cv2.moments(contour)
                # if moments["m00"] == 0:
                #     continue

                # center_x = moments["m10"] / moments["m00"]
                # center_y = moments["m01"] / moments["m00"]

                targets.append(
                    TargetDetection(
                        colour,
                        center_x,
                        center_y,
                        circularity,
                        fill_ratio,
                        area,
                        contour,
                    )
                )

        return targets

    def get_best_target(self, targets: List[TargetDetection]) -> TargetDetection | None:
        best_target = None
        min_dist = float("inf")

        center_x = 250
        center_y = 250

        for target in targets:
            if not self.is_valid_target(target):
                continue
            dist = (center_x - target.x) ** 2 + (center_y - target.y) ** 2
            if dist < min_dist:
                min_dist = dist
                best_target = target

        return best_target

    def is_valid_target(self, target: TargetDetection) -> bool:
        # Thresholds for detection
        MIN_AREA = 300  # Minimum contour area in pixels
        MIN_CIRCULARITY = 0.6  # Circularity threshold (1.0 = perfect circle)
        MIN_FILL_RATIO = 0.7  # Minimum ratio of colored pixels to contour area (1.0 = completely filled)
        return not (
            target.area < MIN_AREA
            or target.circularity < MIN_CIRCULARITY
            or target.coverage < MIN_FILL_RATIO
        )
