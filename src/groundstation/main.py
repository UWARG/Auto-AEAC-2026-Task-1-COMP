"""
TCP target parser for groundstation

This module receives newline-delimited CSV target messages over a TCP socket
and logs them as readable strings. Each message contains 2 chars and 2 doubles
separated by commas. The first character indicates the wall's cardinal
direction, the second indicates the color of the target. The doubles
indicate the position up and to the right of the bottom left corner of the
wall respectively.
Cardinal DIRECTIONS dictionary:
n for north, s for south, e for east, w for west
COLORS dictionary:
r for red, g for green, b for black, u for blue, y for yellow, w for white

"""

import logging
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path
import sys
from groundstation.target_receiver import TargetReceiver
from util import GROUNDSTATION_TCP_HOST, GROUNDSTATION_TCP_PORT

COLORS = {
    "r": "red",
    "b": "black",
    "u": "blue",
    "y": "yellow",
    "g": "green",
    "w": "white",
}
DIRECTIONS = {"n": "north", "s": "south", "w": "west", "e": "east"}
TESTING_MODULE = "groundstation.message_testing"


def test() -> None:
    """
    For running the testing module
    """
    subprocess.run([f"{sys.executable}", "-m", TESTING_MODULE])


def main() -> None:
    """
    Main function
    """
    # loggers
    main_logger = logging.getLogger("Main")
    target_info_logger = logging.getLogger("Target_Info")

    # make directory
    logs_folder = Path("logs")
    logs_folder.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_handler_path = logs_folder / f"main_{timestamp}.log"
    main_handler = logging.FileHandler(str(main_handler_path), mode="w")
    target_info_handler_path = logs_folder / f"Task_1_WARG_targets_{timestamp}.txt"
    target_info_handler = logging.FileHandler(str(target_info_handler_path), mode="w")
    formatter = logging.Formatter(
        fmt="%(asctime)s: [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    # adding handlers and formatters
    main_handler.setFormatter(formatter)
    main_logger.addHandler(main_handler)
    target_info_logger.addHandler(target_info_handler)
    # changing logger levels
    main_logger.setLevel(logging.INFO)
    target_info_logger.setLevel(logging.INFO)
    num_corrupted = 0
    assert main_logger is not None
    receiver = TargetReceiver(
        GROUNDSTATION_TCP_HOST, GROUNDSTATION_TCP_PORT, main_logger
    )

    while True:
        try:
            success, info = receiver.get_message()
            if not success or info is None:
                continue
            # direction
            direction = DIRECTIONS[info[0]]
            # color
            color = COLORS[info[1]]
            # amount up and to the right of bottom left corner of wall
            up = float(info[2])
            right = float(info[3])
            # Log a human readable message
            target_info_logger.info(
                f"The {color} target is {up} meters up and {right} meters"
                f" right from the bottom left corner of the {direction}"
                f" wall"
            )
        except KeyError as k:
            num_corrupted += 1
            main_logger.error(f"Key Error, {k}")
        except ValueError as v:
            num_corrupted += 1
            main_logger.error(f"Invalid double value, {v}")
        except KeyboardInterrupt:
            main_logger.info(
                f"Terminating Program, number of corrupted" f" messages:{num_corrupted}"
            )
            # Flush all handlers to ensure data is written to disk
            for handler in main_logger.handlers:
                handler.flush()
            for handler in target_info_logger.handlers:
                handler.flush()
            break
        except Exception as e:
            main_logger.error(f"A problem occured {e}")

    receiver.close()


if __name__ == "__main__":
    # The self-test harness (which streams fake targets at the receiver) only
    # runs when explicitly requested with --test; by default the groundstation
    # just listens for the real airside connection.
    if "--test" in sys.argv:
        multiprocessing.Process(target=test).start()
    main()
