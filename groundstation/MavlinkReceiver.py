import time
import logging
from pymavlink import mavutil

"""
This class decodes Mavlink StatusText messages and returns them
as a string array

"""


class MavlinkReciever:

    def __init__(self, connection_string: str) -> mavutil.mavlink_connection:
        while not self.__mavlink_connected(connection_string):
            time.sleep(1)

    def __mavlink_connected(self, connection_string: str) -> bool:
        try:
            self.connection = mavutil.mavlink_connection(connection_string)
            logging.info(
                f"Connected to {self.connection.target_system}"
                f", component {self.connection.target_component}"
            )
            self.connection.wait_heartbeat()
            logging.info("Received heartbeat")
            return True
        except Exception as e:
            logging.error(f"Encountered Error {e}. Trying Again")
            return False

    def get_message(self) -> tuple[True, list[str]] | tuple[bool, None]:
        msg = self.connection.recv_match(type="STATUSTEXT", blocking=True)
        if not msg:
            logging.error("Recieved empty message")
            return False, None
        if msg.get_type() == "BAD_DATA":
            logging.error("Received bad data")
            return False, None
        try:
            text = msg.text
            if isinstance(text, bytes):
                message = text.decode("UTF-8").strip("\x00")
                info = message.split(",")
                return True, info
            else:
                logging.error(f"Invalid format, {text}")
        except Exception as e:
            logging.error(f"message processing failed {e}")
        return False, None
