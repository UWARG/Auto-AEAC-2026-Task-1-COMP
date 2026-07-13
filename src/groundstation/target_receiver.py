"""TCP target receiver for the groundstation.

Listens as a TCP server for the airside system, which connects as a client and
streams newline-delimited CSV target messages (``direction,colour,up,right``).
Replaces the previous MAVLink STATUSTEXT receiver.
"""

import logging
import socket


class TargetReceiver:
    """Accepts a TCP connection from airside and yields parsed target lines."""

    def __init__(self, host: str, port: int, logger: logging.Logger) -> None:
        self.logger = logger
        self.host = host
        self.port = port

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.logger.info(f"Groundstation listening on {host}:{port}")

        self.connection: socket.socket | None = None
        self.buffer = ""

    def __accept(self) -> None:
        """Block until an airside client connects."""
        self.logger.info("Waiting for airside connection...")
        self.connection, address = self.server.accept()
        self.buffer = ""
        self.logger.info(f"Airside connected from {address}")

    def get_message(self) -> "tuple[bool, list[str]] | tuple[bool, None]":
        """Return (True, [direction, colour, up, right]) for the next message.

        Blocks until a full line is available. Returns (False, None) on a
        malformed line; transparently re-accepts if the airside client
        disconnects.
        """
        if self.connection is None:
            self.__accept()

        # Read until we have a complete newline-terminated line.
        while "\n" not in self.buffer:
            try:
                chunk = self.connection.recv(1024)  # type: ignore[union-attr]
            except OSError as e:
                self.logger.error(f"Socket error while receiving: {e}")
                self.connection = None
                return False, None

            if not chunk:
                # Peer closed the connection; wait for a new client.
                self.logger.info("Airside disconnected, awaiting new connection")
                self.connection = None
                self.__accept()
                continue
            self.buffer += chunk.decode("utf-8", errors="replace")

        line, self.buffer = self.buffer.split("\n", 1)
        line = line.strip()
        if not line:
            return False, None

        info = [part.strip() for part in line.split(",")]
        if len(info) != 4:
            self.logger.error(f"Malformed message (expected 4 fields): {line!r}")
            return False, None

        return True, info

    def close(self) -> None:
        """Close the active connection and the listening server socket."""
        if self.connection is not None:
            try:
                self.connection.close()
            except OSError:
                pass
            self.connection = None
        try:
            self.server.close()
        except OSError:
            pass
