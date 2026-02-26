import time
import random
import logging
import datetime
from pathlib import Path
import groundstation.main as groundscript
from pymavlink import mavutil

CONNECTION_STRING = "tcp:localhost:1400"
AIRSIDE_COMPONENT_ID = 191
NUM_MESSAGES = 5


def main() -> None:
    connection = mavutil.mavlink_connection(
        CONNECTION_STRING, source_system=1, source_component=AIRSIDE_COMPONENT_ID
    )
    time.sleep(1)
    connection.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=Path("logs", f"message_logs_{timestamp}.log"), level=logging.INFO
    )
    message_logger = logging.getLogger("Message_Logger")
    for _ in range(NUM_MESSAGES):
        message = (
            f"{random.choice(list(groundscript.DIRECTIONS.keys()))},"
            f"{random.choice(list(groundscript.COLORS.keys()))},"
            f"{1+random.random()*9},"
            f"{1+random.random()*9}"
        )
        message_logger.info(f"{message}")
        encoded_message = message.encode()
        connection.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_INFO, encoded_message
        )

    return


if __name__ == "__main__":
    main()
    print("done")
