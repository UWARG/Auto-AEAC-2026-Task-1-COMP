"""
MAVLink communication interface for drone control.

This module provides a MavlinkComm class that handles MAVLink communication with a drone,
including position tracking, RC channel monitoring, and data stream management.
"""

from pymavlink import mavutil
from typing import Any
import socket
from util import (
    AIRSIDE_COMPONENT_ID,
    UINT16_MAX,
    RCChannel,
    MavlinkMessageType,
    MappedTarget,
    Direction,
    GROUNDSTATION_TCP_HOST,
    GROUNDSTATION_TCP_PORT,
    serialize_mapped_target,
)
import logging
import time

SERIAL_PORT = "/dev/ttyAMA0"

GROUNDSTATION_CONNECT_WAIT_SEC = 60.0
GROUNDSTATION_CONNECT_RETRY_SEC = 2.0


class MavlinkComm:
    """Airside communication hub.

    Reports mapped targets to the groundstation over a TCP socket. Optionally
    uses a MAVLink serial link to the drone for heading and the RC-switch
    post-processing trigger; this link can be disabled entirely (``use_mavlink
    = False``) so the system runs standalone with a terminal trigger.
    """

    def __init__(
        self,
        logger: logging.Logger,
        use_mavlink: bool = True,
        post_processing_request_channel: int = 11,
        use_socket: bool = True,
        groundstation_host: str = GROUNDSTATION_TCP_HOST,
        groundstation_port: int = GROUNDSTATION_TCP_PORT,
    ) -> None:
        """Initialize drone connection and the groundstation target socket."""
        self.logger = logger
        self.use_mavlink = use_mavlink
        self.post_processing_request_channel = post_processing_request_channel
        # heading in degrees
        self.heading: float | None = None

        self.rc_channels: dict[int, RCChannel] = {
            i: RCChannel(channel=i, raw=0, is_active=False)
            for i in range(0, 20)  # use 11 to 14 inclusive tho
        }

        self.post_processing_requested_flag = False
        self.mav: Any | None = None

        # Groundstation target-reporting socket (independent of the drone link).
        self.use_socket = use_socket
        self.groundstation_host = groundstation_host
        self.groundstation_port = groundstation_port
        self.socket: socket.socket | None = None
        if self.use_socket:
            self.__connect_groundstation()

        if self.use_mavlink:
            while not self.__mavlink_connect():
                self.logger.info("Failed to connect to drone, retrying...")
                time.sleep(1)

                while not self.__request_data_streams():
                    self.logger.error("Failed to request data streams, retrying...")
                    time.sleep(1)

    def __connect_groundstation(self) -> bool:
        """Open the TCP connection to the groundstation target receiver.

        Best-effort single attempt: failure is non-fatal so the airside system
        keeps running (and capturing data) even if the groundstation is not up
        yet.
        """
        try:
            sock = socket.create_connection(
                (self.groundstation_host, self.groundstation_port), timeout=5
            )
        except OSError as e:
            self.logger.warning(
                f"Could not connect to groundstation at "
                f"{self.groundstation_host}:{self.groundstation_port}: {e}"
            )
            self.socket = None
            return False

        self.socket = sock
        self.logger.info(
            f"Connected to groundstation at "
            f"{self.groundstation_host}:{self.groundstation_port}"
        )
        return True

    def _ensure_groundstation_connection(
        self, wait_timeout: float = GROUNDSTATION_CONNECT_WAIT_SEC
    ) -> bool:
        """Ensure the groundstation socket is open, waiting/retrying if needed.

        Targets are only sent at the end of the run, so this gives the operator
        a window to start the groundstation. Returns False if no connection
        could be made within ``wait_timeout`` seconds.
        """
        if self.socket is not None:
            return True

        deadline = time.monotonic() + wait_timeout
        while True:
            if self.__connect_groundstation():
                return True
            if time.monotonic() >= deadline:
                return False
            self.logger.info(
                f"Waiting for groundstation at {self.groundstation_host}:"
                f"{self.groundstation_port} (start it now)..."
            )
            time.sleep(GROUNDSTATION_CONNECT_RETRY_SEC)

    def __mavlink_connect(self) -> bool:
        """Establish MAVLink connection to drone via serial port."""
        try:
            self.mav = mavutil.mavlink_connection(
                SERIAL_PORT,
                baud=57600,
                source_component=AIRSIDE_COMPONENT_ID,
                source_system=1,
            )

            self.mav.wait_heartbeat()
            self.logger.info(
                f"Heartbeat received from system {self.mav.target_system}, component {self.mav.target_component}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to drone: {e}")
            return False

        self.logger.info("Connected to drone")
        return True

    def __request_data_streams(self) -> bool:
        """Request position and RC channel data streams from drone."""
        try:
            # Request position data at 1 Hz
            self.mav.mav.request_data_stream_send(
                self.mav.target_system,  # Target system ID (the drone)
                self.mav.target_component,  # Target component ID (autopilot)
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,  # Position data stream
                1,  # Rate: 1 Hz
                1,  # Start streaming (1=enable, 0=disable)
            )

            # Request RC channel data at 5 Hz
            self.mav.mav.request_data_stream_send(
                self.mav.target_system,  # Target system ID (the drone)
                self.mav.target_component,  # Target component ID (autopilot)
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,  # RC channels data stream
                5,  # Rate: 5 Hz
                1,  # Start streaming (1=enable, 0=disable)
            )

            self.logger.info("Requested GLOBAL_POSITION_INT and RC_CHANNELS streams")
        except Exception as e:
            self.logger.error(f"Failed to request data streams: {e}")
            return False

        return True

    def process_data_stream(self) -> bool:
        """Process incoming MAVLink messages and update drone state."""
        if not self.use_mavlink:
            self.logger.warning("Mavlink is disabled, skipping processing data stream")
            return False
        if self.mav is None:
            self.logger.warning("Mavlink connection is unavailable")
            return False
        msg = self.mav.recv_match(
            type=[m.value for m in MavlinkMessageType],
            blocking=False,
        )
        if msg is None:
            return False

        if msg.get_type() == MavlinkMessageType.GLOBAL_POSITION_INT.value:
            self.logger.info(f"Received GLOBAL_POSITION_INT: {msg}")

            # Extract heading from hdg field (in centidegrees, convert to degrees)
            if msg.hdg is not None and msg.hdg != UINT16_MAX:
                self.heading = msg.hdg / 100.0

            return True

        elif msg.get_type() == MavlinkMessageType.RC_CHANNELS.value:
            self.logger.info(f"Received RC_CHANNELS: {msg}")

            # Update RC channels by reading 'chanX_raw' attributes from message
            for rc_channel_num in self.rc_channels.keys():
                attr_name = f"chan{rc_channel_num}_raw"
                if not hasattr(msg, attr_name):
                    continue
                raw = getattr(msg, attr_name)
                raw = raw if raw is not None else 0
                self.rc_channels[rc_channel_num] = RCChannel(
                    channel=rc_channel_num, raw=raw, is_active=raw >= 1200
                )

                if (
                    rc_channel_num is self.post_processing_request_channel
                    and raw >= 1200
                ):
                    self.post_processing_requested_flag = True

            return True

        return False

    def _heading_to_direction(self, heading: float) -> Direction:
        """Convert heading in degrees to cardinal direction."""
        if 45 <= heading < 135:
            return Direction.EAST
        elif 135 <= heading < 225:
            return Direction.SOUTH
        elif 225 <= heading < 315:
            return Direction.WEST
        else:
            return Direction.NORTH

    def get_heading_direction(self) -> Direction:
        """
        Get current drone heading as a cardinal direction.
        Return NORTH if heading is unavailable.
        """
        if not self.use_mavlink:
            self.logger.warning(
                "Mavlink is disabled, returning default heading direction NORTH"
            )
            return Direction.NORTH
        if self.heading is None:
            self.logger.warning("Position is not available")
            return Direction.NORTH
        return self._heading_to_direction(self.heading)

    def get_rc_channel(self, channel: int) -> RCChannel:
        """Get RC channel data for specified channel number."""
        if not self.use_mavlink:
            self.logger.warning(
                "Mavlink is disabled, returning default RC channel data"
            )
            return RCChannel(channel=channel, raw=0, is_active=False)
        if channel not in self.rc_channels:
            self.logger.warning(f"Channel {channel} is not available")
            return RCChannel(channel=channel, raw=0, is_active=False)
        return self.rc_channels[channel]

    def get_heading(self) -> float:
        """Get current drone heading in degrees, returns 0 if unavailable."""
        if not self.use_mavlink:
            self.logger.warning("Mavlink is disabled, returning default heading 0.0")
            return 0.0
        if self.heading is None:
            self.logger.warning("Heading is not available")
            return 0.0
        return self.heading

    def send_mapped_target(self, mapped_target: MappedTarget) -> bool:
        """Send a mapped target to the groundstation over the TCP socket.

        Returns True only if the message was actually written to the socket.
        Waits for the groundstation to become available (see
        ``_ensure_groundstation_connection``) and retries once on a broken
        connection.
        """
        if not self.use_socket:
            self.logger.warning(
                f"Groundstation socket disabled, would have sent target: {mapped_target}"
            )
            return False

        message = serialize_mapped_target(mapped_target) + "\n"
        # Two passes: one to (re)connect-and-send, one more if the existing
        # socket turns out to be broken when we try to write to it.
        for _ in range(2):
            if not self._ensure_groundstation_connection():
                self.logger.error(
                    "Groundstation connection is unavailable, cannot send target"
                )
                return False
            try:
                assert self.socket is not None
                self.socket.sendall(message.encode("utf-8"))
                return True
            except OSError as e:
                self.logger.error(f"Failed to send target to groundstation: {e}")
                # Drop the broken socket and let the next pass reconnect.
                self.close()

        self.logger.error("Failed to send target to groundstation after retry")
        return False

    def post_processing_requested(self) -> bool:
        return self.post_processing_requested_flag

    def close(self) -> None:
        """Close the groundstation socket (and any drone link resources)."""
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None
