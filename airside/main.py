"""Main control loop for airside system."""

import logging

from airside.detection.oakd import OakD
from airside.post_processing import post_processing
from util import Coordinate, Vector3d
from mavlink_comm import MavlinkComm

def main() -> None:
    """Main control loop for airside drone operations."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting airside...")

    mav_comm = MavlinkComm()

    oakd = OakD()

    while mav_comm.post_processing_requested() is False:
        mav_comm.process_data_stream()

        oakd.run()
    
    # TODO: Feed in the correct database and targets files
    post_processing.run("rtab-data.db", "targets.txt", mav_comm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, exiting gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
        raise
    finally:
        logging.info("All operations complete, shutting down.")
