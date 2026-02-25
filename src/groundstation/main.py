"""
Mavlink statustext parser for groundstation

This module recieves a statustext message and logs it as a readable string.
The message will be an encoded byte which contains 2 chars and 2 doubles
seperated by commas. The first character indicates the wall's cardinal
direction, the second indicates the color of the target. The doubles
indicate the position up and to the right of the bottom left corner of the
wall respectively.
Cardinal DIRECTIONS dictionary:
n for north, s for south, e for east, w for west
COLORS dictionary:
r for red, g for green, b for black, u for blue, y for yellow

"""

import logging
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path
import sys
from groundstation.MavlinkReceiver import MavlinkReciever

CONNECTION_STRING = "tcpin:localhost:1400"
COLORS = {"r": "red", "b": "black", "u": "blue", "y": "yellow", "g": "green"}
DIRECTIONS = {"n": "north", "s": "south", "w": "west", "e": "east"}
TESTING_MODULE = "groundstation.message_test"


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
    Path("logs").mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_handler_path = Path("logs", f"main_{timestamp}.log")
    main_handler = logging.FileHandler(main_handler_path, mode="w")
    target_info_handler_path = Path("logs", f"Task_1_WARG_targets_{timestamp}.txt")
    target_info_handler = logging.FileHandler(target_info_handler_path, mode="w")
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
    receiver = MavlinkReciever(CONNECTION_STRING, main_logger)

    while True:
        try:
            success, info = receiver.get_message()
            if not success:
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
            break
        except Exception as e:
            main_logger.error(f"A problem occured {e}")


if __name__ == "__main__":
    multiprocessing.Process(target=test).start()
    main()
    print("done")
