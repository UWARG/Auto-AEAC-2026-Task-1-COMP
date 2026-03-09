import threading
import logging
import time

from .abstract_camera import AbstractCamera
from util import MappedTarget, Direction, Coordinate, Colours


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

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            mapped_target = MappedTarget(
                colour=Colours.GREEN,
                location=Coordinate(1.0, 2.0, 3.0),
                direction=Direction.NORTH,
                wall_target=False,
            )

            if mapped_target is not None:
                self.detections_logger.info(mapped_target)
                self.detailed_detections_logger.info(mapped_target)
                self.main_logger.info(f"Detected target: {mapped_target}")

        self.main_logger.info("Stopping Arducam thread.")
