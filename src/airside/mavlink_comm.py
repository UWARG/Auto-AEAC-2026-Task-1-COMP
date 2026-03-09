"""
MAVLink communication interface for drone control.

This module provides a MavlinkComm class that handles MAVLink communication with a drone,
including position tracking, RC channel monitoring, and data stream management.
"""

from pymavlink import mavutil
from typing import Any
from util import (
    AIRSIDE_COMPONENT_ID,
    UINT16_MAX,
    RCChannel,
    MavlinkMessageType,
    MappedTarget,
    Direction,
)
import logging
import time

SERIAL_PORT = "/dev/ttyAMA0"


class MavlinkComm:
    """Handles MAVLink communication and data processing for drone control."""

    def __init__(self, logger: logging.Logger, use_mavlink: bool = True, post_processing_request_channel: int = 11) -> None:
        """Initialize drone connection and request data streams."""
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

        if self.use_mavlink:
            while not self.__mavlink_connect():
                self.logger.info("Failed to connect to drone, retrying...")
                time.sleep(1)

                while not self.__request_data_streams():
                    self.logger.error("Failed to request data streams, retrying...")
                    time.sleep(1)

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

                if rc_channel_num is self.post_processing_request_channel and raw >= 1200:
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
            self.logger.warning("Mavlink is disabled, returning default heading direction NORTH")
            return Direction.NORTH
        if self.heading is None:
            self.logger.warning("Position is not available")
            return Direction.NORTH
        return self._heading_to_direction(self.heading)

    def get_rc_channel(self, channel: int) -> RCChannel:
        """Get RC channel data for specified channel number."""
        if not self.use_mavlink:
            self.logger.warning("Mavlink is disabled, returning default RC channel data")
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

    def send_mapped_target(self, mapped_target: MappedTarget, attempt: int = 0) -> None:
        """Send target to ground station."""
        if not self.use_mavlink:
            self.logger.warning(f"Mavlink is disabled, would have sent target: {mapped_target}")
            return
        if self.mav is None:
            self.logger.error("Mavlink connection is unavailable, cannot send target")
            return
        if attempt > 3:
            self.logger.error("Failed to send target to ground after 3 attempts")
            return

        try:
            self.mav.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_INFO,  # Severity: informational
                f"{mapped_target}".encode(),  # Convert string to bytes
            )
        except Exception as e:
            self.logger.error(f"Failed to send target to ground: {e}")
            self.send_mapped_target(mapped_target, attempt + 1)

    def post_processing_requested(self) -> bool:
        return self.post_processing_requested_flag
