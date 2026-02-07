"""Main control loop for airside system."""

import logging
import queue
import threading

from airside.detection.oakd import OakD
from airside.detection.arducam import Arducam
from airside.post_processing import post_processing
from util import Coordinate, MappedTarget, Vector3d
from mavlink_comm import MavlinkComm

def main() -> None:
    """Main control loop for airside drone operations."""
    main_logger.info("Starting airside...")

    detections_logger = logging.getLogger("detections")

    mav_comm = MavlinkComm(main_logger)

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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        main()
    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt received, exiting gracefully...")
    except Exception as e:
        main_logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        raise
    finally:
        main_logger.info("All operations complete, shutting down.")
