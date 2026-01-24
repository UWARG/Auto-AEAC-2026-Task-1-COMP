import logging
from groundstation.MavlinkReceiver import MavlinkReciever


CONNECTION_STRING = "/dev/ttyAMA0"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    receiver = MavlinkReciever(CONNECTION_STRING)
    try:
        while True:
            success, info = receiver.get_message()
            if not success:
                continue
            # direction
            if info[0] == "n":
                direction = "North"
            elif info[0] == "s":
                direction = "South"
            elif info[0] == "w":
                direction = "West"
            elif info[0] == "e":
                direction = "East"
            # color
            if info[1] == "r":
                color = "red"
            elif info[1] == "b":
                color = "blue"
            elif info[1] == "y":
                color = "yellow"
            elif info[1] == "g":
                color = "green"
            # amount up and to the right of bottom left corner of wall
            up = info[2]
            right = info[3]
            # Log a human readable message
            logging.info(
                f"The {color} target is {up} units up and {right} units"
                f" right from the bottom left corner of the {direction}"
                f" wall"
            )

    except IndexError:
        logging.error("Error in Message's data")
    except KeyboardInterrupt:
        print("Terminating Program")


if __name__ == "__main__":
    main()
