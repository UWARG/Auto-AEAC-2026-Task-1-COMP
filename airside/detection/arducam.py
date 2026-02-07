import threading
import logging

from util import MappedTarget, Direction, Coordinate, Colour

class Arducam(threading.Thread):
    def __init__(self, main_logger: logging.Logger, detections_logger: logging.Logger, stop_event: threading.Event) -> None:
        super().__init__()
        self.main_logger = main_logger
        self.detections_logger = detections_logger
        self.stop_event = stop_event

    def stop(self):
        pass

    def run(self):
        while not self.stop_event.is_set():
            mapped_target = MappedTarget(
                colour=Colour.GREEN, location=Coordinate(1.0, 2.0, 3.0), direction=Direction.NORTH
            )
            
            if mapped_target is not None:
                self.detections_logger.info(mapped_target)
                self.main_logger.info(f"Detected target: {mapped_target}")

        self.main_logger.info("Stopping Arducam thread.")
        self.stop()
