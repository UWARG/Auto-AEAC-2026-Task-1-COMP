"""Main control loop for airside system."""

import logging
import os
import time
import threading

from airside.detection.oakd import OakD
from airside.detection.arducam import Arducam
from airside.post_processing import post_processing
from util import Coordinate, MappedTarget, Vector3d
from mavlink_comm import MavlinkComm


def main() -> None:
    """Main control loop for airside arch."""
    main_logger.info("Starting airside...")

    mav_comm = MavlinkComm(main_logger)

    detections_logger = logging.getLogger("detections")
    detections_logger.setLevel(logging.INFO)
    detections_logger.propagate = False

    detections_formatter = logging.Formatter("%(message)s")

    detections_handler_file = logging.FileHandler("targets.txt")
    detections_handler_file.setFormatter(detections_formatter)
    detections_logger.addHandler(detections_handler_file)

    stop_event = threading.Event()

    oakd = OakD(main_logger, detections_logger, stop_event)
    arducam = Arducam(main_logger, detections_logger, stop_event)

    oakd.start()
    arducam.start()

    while mav_comm.post_processing_requested() is False:
        mav_comm.process_data_stream()

    stop_event.set()
    oakd.join()
    arducam.join()

    post_processing.run("rtab-data.db", "targets.txt", mav_comm)


if __name__ == "__main__":
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False

    main_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    main_handler_console = logging.StreamHandler()
    main_handler_console.setFormatter(main_formatter)
    main_logger.addHandler(main_handler_console)

    os.makedirs("logs", exist_ok=True)
    main_logger_handler_file = logging.FileHandler(
        f"logs/airside_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.log"
    )
    main_logger_handler_file.setFormatter(main_formatter)
    main_logger.addHandler(main_logger_handler_file)

    try:
        main()
    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt received, exiting gracefully...")
    except Exception as e:
        main_logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        raise
    finally:
        main_logger.info("All operations complete, shutting down.")
