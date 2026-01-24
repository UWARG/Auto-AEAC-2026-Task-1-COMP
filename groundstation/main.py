from pymavlink import mavutil
from common.modules.logger import logger
from common.modules.logger import logger_main_setup
from common.modules.read_yaml import read_yaml

CONNECTION_STRING = "tcp.localhost:1400"


def main() -> None:
    success, config = read_yaml.open_config(logger.CONFIG_FILE_PATH)
    if not success:
        print("Could not read config file")
    assert success is True
    success, main_logger, _ = logger_main_setup.setup_main_logger(config)
    if not success:
        print("Could not create main_logger")
    assert success is True

    connection = mavutil.mavlink_connection(CONNECTION_STRING)
    connection.wait_heartbeat()
    while True:
        msg = connection.recv_match(type="STATUSTEXT", blocking=True)
        if not msg:
            main_logger.error("Recieved empty message")
            continue
        if msg.get_type() == "BAD_DATA":
            main_logger.error("Recieved bad data")
            continue
        try:
            text = msg.text
            if isinstance(text, bytes):
                message = text.decode("UTF-8").strip("\x00")
                info = message.split(",")
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
                main_logger.info(
                    f"The {color} target is {up} units up and {right} units"
                    f" right from the bottom left corner of the {direction}"
                    f" wall"
                )

        except AttributeError:
            main_logger.error('Could not find property "text"')
        except UnicodeDecodeError:
            main_logger.error("Could not decode text")
        except IndexError:
            main_logger.error("Error in message's data")


if __name__ == "__main__":
    main()
