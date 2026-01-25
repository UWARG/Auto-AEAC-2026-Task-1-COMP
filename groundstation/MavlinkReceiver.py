import time
import logging
from pymavlink import mavutil

"""
This class decodes Mavlink StatusText messages and returns them
as a string array

"""


class MavlinkReciever:
    # airside component id
    AIRSIDE_COMPONENT_ID = 191

    def __init__(
        self, connection_string: str, logger: logging.Logger
    ) -> mavutil.mavlink_connection:
        while not self.__mavlink_connected(connection_string):
            time.sleep(1)
        self.logger = logger

    def __mavlink_connected(self, connection_string: str) -> bool:
        try:
            self.connection = mavutil.mavlink_connection(connection_string)
            self.connection.wait_heartbeat()
            self.logger.info(
                f"Heartbeat recieved from {self.connection.target_system}"
                f", component {self.connection.target_component}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Encountered Error {e}. Trying Again")
            return False

    def get_message(self) -> tuple[True, list[str]] | tuple[bool, None]:
        msg = self.connection.recv_match(type="STATUSTEXT", blocking=True)
        if not msg:
            self.logger.error("Recieved empty message")
            return False, None
        if msg.get_type() == "BAD_DATA":
            self.logger.error("Received bad data")
            return False, None
        if msg.get_srcComponent() != self.AIRSIDE_COMPONENT_ID:
            self.logger.info(f"Ignoring message from {msg.get_srcComponent}")
        try:
            text = msg.text
            if isinstance(text, bytes):
                message = text.decode("UTF-8").strip("\x00")
                info = message.split(",")
                return True, info
            else:
                self.logger.error(f"Invalid format, {text}")
        except Exception as e:
            self.logger.error(f"message processing failed {e}")
        return False, None
