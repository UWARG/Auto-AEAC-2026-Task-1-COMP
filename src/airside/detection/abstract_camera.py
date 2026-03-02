from abc import ABC, abstractmethod
import threading
import logging

from util import MappedTarget, Direction, Coordinate, Colours


class AbstractCamera(threading.Thread, ABC):
    def __init__(
        self,
        main_logger: logging.Logger,
        detections_logger: logging.Logger,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.main_logger = main_logger
        self.detections_logger = detections_logger
        self.stop_event = stop_event

    @abstractmethod
    def run(self):
        pass
