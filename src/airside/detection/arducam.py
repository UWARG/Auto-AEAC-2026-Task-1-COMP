import threading
import logging
import time

from .abstract_camera import AbstractCamera
from util import Colours, Coordinate, Target


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
            # TODO: Add arducam detection code
            target = Target(
                colour=Colours.RED,
                location=Coordinate(0.0, 0.0, 0.0),
            )

            if target is not None:
                self.detections_logger.info(target)
                self.detailed_detections_logger.info(target)
                self.main_logger.info(f"Detected target: {target}")

        self.main_logger.info("Stopping Arducam thread.")
