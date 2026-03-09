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
from airside.mavlink_comm import MavlinkComm


# If set to false, then the system uses the RC switch to trigger post-processing.
# If set to true, then post-processing is triggered by pressing enter in the console.
TRIGGER_DEBUG_MODE = True
USE_MAVLINK = True

POST_PROCESSING_REQUEST_CHANNEL = (
    11  # RC channel number to monitor for post-processing trigger
)

OBSTACLE_PCL_FILENAME = "obstaclePCL.ply"
GROUND_PCL_FILENAME = "groundPCL.ply"


def main(starting_time: str) -> None:
    """Main control loop for airside arch."""
    main_logger.info("Starting airside...")

    output_folder = Path(__file__).parent.parent.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    if USE_MAVLINK:
        main_logger.info(f"Initializing Mavlink connection")
    else:
        main_logger.info(f"Mavlink connection disabled")
    mav_comm = MavlinkComm(main_logger, USE_MAVLINK, POST_PROCESSING_REQUEST_CHANNEL)

    detections_formatter = logging.Formatter("%(message)s")
    main_logger.info("Creating output directories and files")

    detections_logger = logging.getLogger("detections")
    detections_logger.setLevel(logging.INFO)
    detections_logger.propagate = False
    detections_handler_file = logging.FileHandler(
        str(output_folder / f"targets_{starting_time}.txt")
    )
    detections_handler_file.setFormatter(detections_formatter)
    detections_logger.addHandler(detections_handler_file)

    detailed_detections_logger = logging.getLogger("detailed_detections")
    detailed_detections_logger.setLevel(logging.INFO)
    detailed_detections_logger.propagate = False
    detailed_detections_handler_file = logging.FileHandler(
        str(output_folder / f"detailed_targets_{starting_time}.txt")
    )
    detailed_detections_handler_file.setFormatter(detections_formatter)
    detailed_detections_logger.addHandler(detailed_detections_handler_file)

    stop_event = threading.Event()

    obstacle_pcl_path = str(output_folder / OBSTACLE_PCL_FILENAME)
    ground_pcl_path = str(output_folder / GROUND_PCL_FILENAME)

    oakd = OakD(
        main_logger,
        detections_logger,
        detailed_detections_logger,
        stop_event,
        obstacle_pcl_path=obstacle_pcl_path,
        ground_pcl_path=ground_pcl_path,
    )
    # arducam = Arducam(main_logger, detections_logger, detailed_detections_logger, stop_event)

    initial_heading = mav_comm.get_heading_direction()

    oakd.start()
    # arducam.start()

    if TRIGGER_DEBUG_MODE:
        main_logger.info(
            "Trigger debug mode enabled. Press Enter to trigger post-processing."
        )
        input()
    elif USE_MAVLINK:
        while mav_comm.post_processing_requested() is False:
            mav_comm.process_data_stream()

    stop_event.set()
    oakd.join()
    # arducam.join()

    # Flush all handlers to ensure data is written to disk
    for handler in detections_logger.handlers:
        handler.flush()
    for handler in detailed_detections_logger.handlers:
        handler.flush()
    for handler in main_logger.handlers:
        handler.flush()

    post_processing.run(
        obstacle_pcl_path,
        ground_pcl_path,
        str(output_folder / f"targets_{starting_time}.txt"),
        mav_comm,
        initial_heading,
    )


if __name__ == "__main__":
    starting_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False

    main_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    main_handler_console = logging.StreamHandler()
    main_handler_console.setFormatter(main_formatter)
    main_logger.addHandler(main_handler_console)

    logs_folder = Path("logs")
    logs_folder.mkdir(exist_ok=True, parents=True)
    main_logger_handler_file = logging.FileHandler(
        str(logs_folder / f"airside_{starting_time}.log")
    )
    main_logger_handler_file.setFormatter(main_formatter)
    main_logger.addHandler(main_logger_handler_file)

    try:
        main(starting_time)
    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt received, exiting gracefully...")
    except Exception as e:
        main_logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        raise
    finally:
        main_logger.info("All operations complete, shutting down.")
