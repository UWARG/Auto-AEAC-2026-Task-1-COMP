import logging
from groundstation.MavlinkReceiver import MavlinkReciever


CONNECTION_STRING = "tcp:localhost:1400"

"""
Mavlink statustext parser for groundstation

This module recieves a statustext message and logs it as a readable string.
The message will be an encoded byte which contains 2 chars and 2 doubles
seperated by commas. The first character indicates the wall's cardinal
direction, the second indicates the color of the target. The doubles
indicate the position up and to the right of the bottom left corner of the
wall respectively.
Cardinal directions dictionary:
n for north, s for south, e for east, w for west
Colors dictionary:
r for red, g for green, b for black, u for blue, y for yellow

"""


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # dictionary definition
    colors = {"r": "red",
              "b": "black",
              "u": "blue",
              "y": "yellow",
              "g": "green"}
    directions = {"n": "north", "s": "south", "w": "west", "e": "east"}

    receiver = MavlinkReciever(CONNECTION_STRING)

    while True:
        try:
            success, info = receiver.get_message()
            if not success:
                continue
            # direction
            direction = directions[info[0]]
            # color
            color = colors[info[1]]
            # amount up and to the right of bottom left corner of wall
            up = info[2]
            right = info[3]
            # Log a human readable message
            logging.info(
                f"The {color} target is {up} units up and {right} units"
                f" right from the bottom left corner of the {direction}"
                f" wall"
            )
        except KeyError as k:
            logging.error(f"Key Error, {k}")
        except KeyboardInterrupt:
            print("Terminating Program")
            break
        except Exception as e:
            logging.error(f"A problem occured {e}")


if __name__ == "__main__":
    main()
