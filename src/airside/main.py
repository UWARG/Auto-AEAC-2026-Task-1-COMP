"""Main control loop for airside system."""

import logging
import os
from pathlib import Path
import time
import threading

from airside.detection.oakd.oakd import OakD
from airside.detection.arducam import Arducam
from airside.post_processing import post_processing
from util import Coordinate, Direction, MappedTarget, Vector3d
from mavlink_comm import MavlinkComm


# If set to false, then the system uses the RC switch to trigger post-processing.
# If set to true, then post-processing is triggered by pressing enter in the console.
TRIGGER_DEBUG_MODE = True

USE_MAVLINK = False

def main() -> None:
    """Main control loop for airside arch."""
    main_logger.info("Starting airside...")

    output_folder = Path(__file__).parent.parent.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    if USE_MAVLINK:
        mav_comm = MavlinkComm(main_logger)

    detections_logger = logging.getLogger("detections")
    detections_logger.setLevel(logging.INFO)
    detections_logger.propagate = False

    detections_formatter = logging.Formatter("%(message)s")

    os.makedirs("outputs", exist_ok=True)
    detections_handler_file = logging.FileHandler(str(output_folder / "targets.txt"))
    detections_handler_file.setFormatter(detections_formatter)
    detections_logger.addHandler(detections_handler_file)

    stop_event = threading.Event()

    oakd = OakD(main_logger, detections_logger, stop_event)
    # arducam = Arducam(main_logger, detections_logger, stop_event)

    heading_mapping = {
        Direction.NORTH: 0.0,
        Direction.EAST: 90.0,
        Direction.SOUTH: 180.0,
        Direction.WEST: 270.0,
    }
    if USE_MAVLINK:
        initial_heading_deg = mav_comm.get_heading()
    else:
        initial_heading = 0

    initial_heading = Direction.NORTH
    for dir, deg in heading_mapping.items():
        if abs(initial_heading_deg - deg) <= 45:
            initial_heading = dir
            break

    oakd.start()
    # arducam.start()

    if TRIGGER_DEBUG_MODE:
        main_logger.info("Trigger debug mode enabled. Press Enter to trigger post-processing.")
        input()
    elif USE_MAVLINK:
        while mav_comm.post_processing_requested() is False:
            mav_comm.process_data_stream()

    stop_event.set()
    oakd.join()
    # arducam.join()

    post_processing.run(str(output_folder / "building.ply"), str(output_folder / "targets.txt"), mav_comm, initial_heading)

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
