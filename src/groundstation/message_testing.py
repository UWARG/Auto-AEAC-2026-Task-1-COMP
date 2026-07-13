import time
import random
import logging
import datetime
import socket
from pathlib import Path
import groundstation.main as groundscript
from util import GROUNDSTATION_TCP_HOST, GROUNDSTATION_TCP_PORT

NUM_MESSAGES = 5
CONNECT_RETRIES = 10


def _connect() -> socket.socket:
    """Connect to the groundstation server, retrying while it starts up."""
    last_error: OSError | None = None
    for _ in range(CONNECT_RETRIES):
        try:
            return socket.create_connection(
                (GROUNDSTATION_TCP_HOST, GROUNDSTATION_TCP_PORT), timeout=5
            )
        except OSError as e:
            last_error = e
            time.sleep(1)
    raise ConnectionError(
        f"Could not connect to groundstation after {CONNECT_RETRIES} attempts: {last_error}"
    )


def main() -> None:
    connection = _connect()

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
        connection.sendall((message + "\n").encode("utf-8"))
        time.sleep(0.2)

    connection.close()
    return


if __name__ == "__main__":
    main()
    print("done")
