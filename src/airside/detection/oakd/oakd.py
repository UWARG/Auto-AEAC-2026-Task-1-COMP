import threading
import logging
from typing import Any
from pathlib import Path

from airside.detection import FRD_conversion
from airside.detection.oakd.camera_bundle import CameraBundle

from ..abstract_camera import AbstractCamera
from util import (
    Colours,
    MappedTarget,
    Direction,
    Coordinate,
    Colour,
    Quaternion,
    Target,
)
from airside.detection.oakd.Basalt_VIO_RTab import add_basalt_vio_rtab
from airside.detection.oakd.object_tracker import add_object_tracker
import depthai as dai
import numpy as np
import open3d as o3d
from airside.detection.oakd.rerun_node import RerunNode
import time


ENABLE_RERUN = False
# If set to -1, the maximum difference theshold is ignored
MAX_TIMESTAMP_DIFF_SEC = -1


class OakD(AbstractCamera):
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
        
        def _save_pcl(points: np.ndarray | None, output_path: str) -> None:
            cloud = o3d.geometry.PointCloud()

            if points is not None and len(points) > 0:
                cloud.points = o3d.utility.Vector3dVector(points)

            o3d.io.write_point_cloud(output_path, cloud)

        def _extract_points(msg: object) -> np.ndarray | None:
            if hasattr(msg, "getPoints"):
                points = msg.getPoints()  # type: ignore[attr-defined]
                return np.asarray(points, dtype=np.float64)
            if hasattr(msg, "getPointsRGB"):
                points, _ = msg.getPointsRGB()  # type: ignore[attr-defined]
                return np.asarray(points, dtype=np.float64)
            return None

        def _drain_latest(queue: Any) -> object | None:
            latest_msg: object | None = None
            while True:
                msg = queue.tryGet()
                if msg is None:
                    break
                latest_msg = msg
            return latest_msg

        with dai.Pipeline() as pipeline:
            cameraBundle = CameraBundle(pipeline)
            qDetections = add_object_tracker(pipeline, cameraBundle)
            add_basalt_vio_rtab(pipeline, cameraBundle)

            slam = cameraBundle.slam
            qSlamTransform = slam.transform.createOutputQueue(maxSize=1, blocking=False)
            qObstaclePCL = slam.obstaclePCL.createOutputQueue(maxSize=1, blocking=False)
            qGroundPCL = slam.groundPCL.createOutputQueue(maxSize=1, blocking=False)
            if ENABLE_RERUN:
                rerunViewer = RerunNode()
                slam.transform.link(rerunViewer.inputTrans)
                slam.passthroughRect.link(rerunViewer.inputImg)
                slam.occupancyGridMap.link(rerunViewer.inputGrid)
                slam.obstaclePCL.link(rerunViewer.inputObstaclePCL)
                slam.groundPCL.link(rerunViewer.inputGroundPCL)

            pipeline.start()
            self.main_logger.info("Starting Pipeline...")

            transform_timestamp: float | None = None
            quat: dai.Quaterniond | None = None
            trans: dai.Point3d | None = None
            obstacle_points: np.ndarray | None = None
            ground_points: np.ndarray | None = None

            try:
                while not self.stop_event.is_set():
                    try:
                        time.sleep(0.01)

                        transform_msg = qSlamTransform.tryGet()
                        if isinstance(transform_msg, dai.TransformData):
                            quat = transform_msg.getQuaternion()
                            trans_mm = transform_msg.getTranslation()
                            trans = dai.Point3d(trans_mm.x / 1000.0, trans_mm.y / 1000.0, trans_mm.z / 1000.0)
                            raw_timestamp = transform_msg.getTimestamp()
                            transform_timestamp = raw_timestamp.total_seconds()
                            self.main_logger.debug(
                                f"SLAM transform | raw: {raw_timestamp} | seconds: {transform_timestamp} | quat: {quat} | trans: {trans}"
                            )

                        obstacle_msg = _drain_latest(qObstaclePCL)
                        if obstacle_msg is not None:
                            latest_obstacle_points = _extract_points(obstacle_msg)
                            if latest_obstacle_points is not None:
                                obstacle_points = latest_obstacle_points

                        ground_msg = _drain_latest(qGroundPCL)
                        if ground_msg is not None:
                            latest_ground_points = _extract_points(ground_msg)
                            if latest_ground_points is not None:
                                ground_points = latest_ground_points

                        # Get tracker outputs
                        detectionMsg = qDetections.tryGet()
                        if detectionMsg and isinstance(
                            detectionMsg, dai.SpatialImgDetections
                        ):
                            if transform_timestamp is None or quat is None or trans is None:
                                self.main_logger.debug(
                                    f"SLAM not ready, logging {len(detectionMsg.detections)} raw detections"
                                )
                                for detection in detectionMsg.detections:
                                    spatial_coordinates = detection.spatialCoordinates
                                    cam_target_coord = Coordinate(
                                        spatial_coordinates.z / 1000.0,
                                        spatial_coordinates.x / 1000.0,
                                        spatial_coordinates.y / 1000.0,
                                    )
                                    # Log without FRD conversion when SLAM not available
                                    mapped_target = Target(
                                        colour=Colours.RED,
                                        location=cam_target_coord,
                                    )
                                    self.main_logger.info(
                                        f"Result (no SLAM): {mapped_target}"
                                    )
                                    for handler in self.main_logger.handlers:
                                        handler.flush()
                                continue

                            detections_timestamp = (
                                detectionMsg.getTimestamp().total_seconds()
                            )

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

                            for detection in detectionMsg.detections:

                                spatial_coordinates = detection.spatialCoordinates

                                cam_target_coord = Coordinate(
                                    spatial_coordinates.z / 1000.0,
                                    spatial_coordinates.x / 1000.0,
                                    spatial_coordinates.y / 1000.0,
                                )
                                origin_cam_q = Quaternion(
                                    quat.qw, quat.qx, -quat.qy, -quat.qz
                                )
                                origin_cam_coord = Coordinate(trans.x, -trans.y, -trans.z)

                                translated_coordinate = (
                                    FRD_conversion.convert_target_to_FRD(
                                        cam_target_coord,
                                        origin_cam_q,
                                        origin_cam_coord,
                                    )
                                )

                                mapped_target = Target(
                                    colour=Colours.RED,  # TODO: Replace with actual colour
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
                final_obstacle_msg = _drain_latest(qObstaclePCL)
                if final_obstacle_msg is not None:
                    final_obstacle_points = _extract_points(final_obstacle_msg)
                    if final_obstacle_points is not None:
                        obstacle_points = final_obstacle_points

                final_ground_msg = _drain_latest(qGroundPCL)
                if final_ground_msg is not None:
                    final_ground_points = _extract_points(final_ground_msg)
                    if final_ground_points is not None:
                        ground_points = final_ground_points

                slam.saveDatabase()

                _save_pcl(
                    obstacle_points,
                    self.obstacle_pcl_path,
                )
                _save_pcl(ground_points, self.ground_pcl_path)

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
