import threading
import logging
from typing import Any
from pathlib import Path
import time

import cv2
import numpy as np
import depthai as dai
import open3d as o3d

from airside.detection import FRD_conversion
from ..abstract_camera import AbstractCamera
from util import (
    Colours,
    Coordinate,
    Quaternion,
    Target,
)


FPS = 60
WIDTH = 640
HEIGHT = 400

# Cap red-circle detection rate so the host-side OpenCV work does not starve
# RTAB-Map's SLAM backend (which also runs on the host CPU) of cycles.
DETECT_FPS_CAP = 15

# Depth gating for back-projected circles (millimetres).
DEPTH_LOWER_MM = 100
DEPTH_UPPER_MM = 15000

# Red-circle detection tuning.
MIN_CONTOUR_AREA = 200
MIN_CIRCULARITY = 0.6

# If set to -1, the maximum timestamp difference threshold is ignored.
MAX_TIMESTAMP_DIFF_SEC = -1


class OakD(AbstractCamera):
    """
    Self-contained OAK-D pipeline: stereo SLAM + classical red-circle
    detection fused into world-frame targets.
    """

    def __init__(
        self,
        main_logger: logging.Logger,
        detections_logger: logging.Logger,
        detailed_detections_logger: logging.Logger,
        stop_event: threading.Event,
        obstacle_pcl_path: str,
        ground_pcl_path: str,
    ):
        super().__init__(
            main_logger, detections_logger, detailed_detections_logger, stop_event
        )
        self.obstacle_pcl_path = obstacle_pcl_path
        self.ground_pcl_path = ground_pcl_path

    @staticmethod
    def _detect_red_circles(bgr: np.ndarray) -> list[tuple[int, int]]:
        """Return list of (px, py) pixel centres of red circular blobs."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        centres: list[tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter**2) < MIN_CIRCULARITY:  # circularity
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centres.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        return centres

    @staticmethod
    def _save_pcl(points: np.ndarray | None, output_path: str) -> None:
        cloud = o3d.geometry.PointCloud()
        if points is not None and len(points) > 0:
            cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_path, cloud)

    @staticmethod
    def _extract_points(msg: object) -> np.ndarray | None:
        if hasattr(msg, "getPoints"):
            points = msg.getPoints()  # type: ignore[attr-defined]
            return np.asarray(points, dtype=np.float64)
        if hasattr(msg, "getPointsRGB"):
            points, _ = msg.getPointsRGB()  # type: ignore[attr-defined]
            return np.asarray(points, dtype=np.float64)
        return None

    @staticmethod
    def _drain_latest(queue: Any) -> object | None:
        latest_msg: object | None = None
        while True:
            msg = queue.tryGet()
            if msg is None:
                break
            latest_msg = msg
        return latest_msg

    def run(self):
        output_folder = Path(__file__).parent.parent.parent.parent.parent / "outputs"
        db_path = output_folder / "building.db"
        if db_path.exists():
            db_path.unlink()
        pcl_obstacle_path = output_folder / "obstaclePCL.ply"
        if pcl_obstacle_path.exists():
            pcl_obstacle_path.unlink()
        pcl_ground_path = output_folder / "groundPCL.ply"
        if pcl_ground_path.exists():
            pcl_ground_path.unlink()

        with dai.Pipeline() as pipeline:
            # --- Cameras & sensors ------------------------------------------
            camRgb = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A, sensorFps=FPS
            )
            monoLeft = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=FPS
            )
            monoRight = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=FPS
            )
            imu = pipeline.create(dai.node.IMU)
            odom = pipeline.create(dai.node.BasaltVIO)
            slam = pipeline.create(dai.node.RTABMapSLAM)

            slam.setDatabasePath(str(db_path))
            slam.setParams(
                {
                    "RGBD/CreateOccupancyGrid": "true",
                    "Grid/3D": "true",
                    "Rtabmap/SaveWMState": "true",
                    "RGBD/ProximityBySpace": "false",
                }
            )

            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # --- Stereo: one node for SLAM, one for detection depth ---------
            # build() of detection consumers can reconfigure a stereo node's
            # resolution/alignment, so SLAM and detection get separate nodes.
            slamStereo = pipeline.create(dai.node.StereoDepth)
            slamStereo.setExtendedDisparity(False)
            slamStereo.setLeftRightCheck(True)
            slamStereo.setSubpixel(True)
            slamStereo.setRectifyEdgeFillColor(0)
            slamStereo.enableDistortionCorrection(True)
            slamStereo.initialConfig.setLeftRightCheckThreshold(10)
            slamStereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

            # Subpixel disabled here (unneeded for circle localization) to keep
            # on-device depth-engine load off SLAM's stereo node.
            detStereo = pipeline.create(dai.node.StereoDepth)
            detStereo.setExtendedDisparity(False)
            detStereo.setLeftRightCheck(True)
            detStereo.setSubpixel(False)
            detStereo.setRectifyEdgeFillColor(0)
            detStereo.enableDistortionCorrection(True)
            detStereo.initialConfig.setLeftRightCheckThreshold(10)
            detStereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

            leftOut = monoLeft.requestOutput((WIDTH, HEIGHT))
            rightOut = monoRight.requestOutput((WIDTH, HEIGHT))
            leftOut.link(slamStereo.left)
            rightOut.link(slamStereo.right)
            leftOut.link(detStereo.left)
            rightOut.link(detStereo.right)

            slamStereo.syncedLeft.link(odom.left)
            slamStereo.syncedRight.link(odom.right)
            slamStereo.depth.link(slam.depth)
            slamStereo.rectifiedLeft.link(slam.rect)
            imu.out.link(odom.imu)
            odom.transform.link(slam.odom)

            colorOut = camRgb.requestOutput((WIDTH, HEIGHT))

            # --- Output queues ----------------------------------------------
            qSlamTransform = slam.transform.createOutputQueue(
                maxSize=1, blocking=False
            )
            qObstaclePCL = slam.obstaclePCL.createOutputQueue(
                maxSize=1, blocking=False
            )
            qGroundPCL = slam.groundPCL.createOutputQueue(maxSize=1, blocking=False)
            qColor = colorOut.createOutputQueue(maxSize=1, blocking=False)
            qDepth = detStereo.depth.createOutputQueue(maxSize=1, blocking=False)

            pipeline.start()
            self.main_logger.info("Starting Pipeline...")

            transform_timestamp: float | None = None
            quat: dai.Quaterniond | None = None
            trans_mm: dai.Point3d | None = None
            obstacle_points: np.ndarray | None = None
            ground_points: np.ndarray | None = None

            # Detection camera intrinsics (read lazily from device calibration).
            det_fx = det_fy = det_cx = det_cy = 0.0
            intrinsics_set = False
            last_detect_time = 0.0
            detect_interval = 1.0 / DETECT_FPS_CAP

            try:
                while not self.stop_event.is_set():
                    try:
                        time.sleep(0.01)

                        transform_msg = qSlamTransform.tryGet()
                        if isinstance(transform_msg, dai.TransformData):
                            quat = transform_msg.getQuaternion()
                            trans_mm = transform_msg.getTranslation()
                            raw_timestamp = transform_msg.getTimestampDevice()
                            transform_timestamp = raw_timestamp.total_seconds()
                            self.main_logger.debug(
                                f"SLAM transform | raw: {raw_timestamp} | seconds: {transform_timestamp} | quat: {quat} | trans (m): ({trans_mm.x / 1000.0}, {-trans_mm.y / 1000.0}, {-trans_mm.z / 1000.0})"
                            )

                        obstacle_msg = self._drain_latest(qObstaclePCL)
                        if obstacle_msg is not None:
                            latest_obstacle_points = self._extract_points(obstacle_msg)
                            if latest_obstacle_points is not None:
                                obstacle_points = latest_obstacle_points

                        ground_msg = self._drain_latest(qGroundPCL)
                        if ground_msg is not None:
                            latest_ground_points = self._extract_points(ground_msg)
                            if latest_ground_points is not None:
                                ground_points = latest_ground_points

                        # Always drain depth so the queue does not back up.
                        depth_msg = qDepth.tryGet()

                        # Throttle the (host-side) red-circle detection.
                        now = time.monotonic()
                        if now - last_detect_time < detect_interval:
                            continue
                        color_msg = qColor.tryGet()
                        if not isinstance(color_msg, dai.ImgFrame) or not isinstance(
                            depth_msg, dai.ImgFrame
                        ):
                            continue
                        last_detect_time = now

                        if not intrinsics_set:
                            calib = pipeline.getDefaultDevice().readCalibration()
                            m = calib.getCameraIntrinsics(
                                dai.CameraBoardSocket(color_msg.getInstanceNum()),
                                color_msg.getWidth(),
                                color_msg.getHeight(),
                            )
                            det_fx, det_fy = m[0][0], m[1][1]
                            det_cx, det_cy = m[0][2], m[1][2]
                            intrinsics_set = True

                        bgr = color_msg.getCvFrame()
                        depth_mm = depth_msg.getFrame().astype(np.float32)
                        if depth_mm.shape[:2] != bgr.shape[:2]:
                            depth_mm = cv2.resize(
                                depth_mm,
                                (bgr.shape[1], bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        # Back-project each red circle into camera-frame spatial
                        # coordinates (mm, OAK RDF: x right, y down, z forward).
                        spatial_targets: list[tuple[float, float, float]] = []
                        for (px, py) in self._detect_red_circles(bgr):
                            z_mm = float(depth_mm[py, px])
                            if z_mm < DEPTH_LOWER_MM or z_mm > DEPTH_UPPER_MM:
                                continue
                            x_mm = (px - det_cx) * z_mm / det_fx
                            y_mm = (py - det_cy) * z_mm / det_fy
                            spatial_targets.append((x_mm, y_mm, z_mm))

                        if not spatial_targets:
                            continue

                        detections_timestamp = (
                            depth_msg.getTimestampDevice().total_seconds()
                        )

                        # If SLAM has not produced a pose yet, log raw detections
                        # in the camera frame without FRD conversion.
                        if (
                            transform_timestamp is None
                            or quat is None
                            or trans_mm is None
                        ):
                            self.main_logger.debug(
                                f"SLAM not ready, logging {len(spatial_targets)} raw detections"
                            )
                            for (x_mm, y_mm, z_mm) in spatial_targets:
                                cam_target_coord = Coordinate(
                                    z_mm / 1000.0, x_mm / 1000.0, y_mm / 1000.0
                                )
                                raw_target = Target(
                                    colour=Colours.RED, location=cam_target_coord
                                )
                                self.main_logger.info(f"Result (no SLAM): {raw_target}")
                                for handler in self.main_logger.handlers:
                                    handler.flush()
                            continue

                        self.main_logger.debug(
                            f"Timestamp comparison | detection: {detections_timestamp:.6f}s | transform: {transform_timestamp:.6f}s | diff: {abs(detections_timestamp - transform_timestamp):.6f}s"
                        )

                        if (
                            MAX_TIMESTAMP_DIFF_SEC != -1
                            and abs(detections_timestamp - transform_timestamp)
                            > MAX_TIMESTAMP_DIFF_SEC
                        ):
                            self.main_logger.debug(
                                f"Skipping detection: timestamps out of sync (difference: {abs(detections_timestamp - transform_timestamp)}, detection: {detections_timestamp}, transform: {transform_timestamp})"
                            )
                            continue

                        for (x_mm, y_mm, z_mm) in spatial_targets:
                            cam_target_coord = Coordinate(
                                z_mm / 1000.0, x_mm / 1000.0, y_mm / 1000.0
                            )
                            origin_cam_q = Quaternion(
                                quat.qw, quat.qx, -quat.qy, -quat.qz
                            )
                            origin_cam_coord = Coordinate(
                                trans_mm.x / 1000.0,
                                -trans_mm.y / 1000.0,
                                -trans_mm.z / 1000.0,
                            )

                            translated_coordinate = FRD_conversion.convert_target_to_FRD(
                                cam_target_coord,
                                origin_cam_q,
                                origin_cam_coord,
                            )

                            mapped_target = Target(
                                colour=Colours.RED,
                                location=translated_coordinate,
                            )

                            self.detections_logger.info(mapped_target)
                            for handler in self.detections_logger.handlers:
                                handler.flush()

                            self.detailed_detections_logger.info(
                                f"Result: {mapped_target} | Pose: {origin_cam_coord} - {origin_cam_q} | Detection: {cam_target_coord}"
                            )
                            for handler in self.detailed_detections_logger.handlers:
                                handler.flush()

                            self.main_logger.info(
                                f"Result: {mapped_target} | Pose: {origin_cam_coord} - {origin_cam_q} | Detection: {cam_target_coord}"
                                f" | SLAM transform timestamp: {transform_timestamp} | Detection timestamp: {detections_timestamp}"
                            )
                            for handler in self.main_logger.handlers:
                                handler.flush()
                    except Exception as e:
                        self.main_logger.error(
                            f"Error in detection loop: {e}", exc_info=True
                        )
                        for handler in self.main_logger.handlers:
                            handler.flush()
                        # Continue running despite errors
                        continue
            finally:
                final_obstacle_msg = self._drain_latest(qObstaclePCL)
                if final_obstacle_msg is not None:
                    final_obstacle_points = self._extract_points(final_obstacle_msg)
                    if final_obstacle_points is not None:
                        obstacle_points = final_obstacle_points

                final_ground_msg = self._drain_latest(qGroundPCL)
                if final_ground_msg is not None:
                    final_ground_points = self._extract_points(final_ground_msg)
                    if final_ground_points is not None:
                        ground_points = final_ground_points

                slam.saveDatabase()

                self._save_pcl(obstacle_points, self.obstacle_pcl_path)
                self._save_pcl(ground_points, self.ground_pcl_path)

                obstacle_count = 0 if obstacle_points is None else len(obstacle_points)
                ground_count = 0 if ground_points is None else len(ground_points)
                self.main_logger.info(
                    f"Saved SLAM point clouds | obstacle={obstacle_count} points, ground={ground_count} points"
                )
                for handler in self.main_logger.handlers:
                    handler.flush()

                pipeline.stop()
        self.main_logger.info("Stopping OakD thread.")
        for handler in self.main_logger.handlers:
            handler.flush()
