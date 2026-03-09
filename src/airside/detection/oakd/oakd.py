import threading
import logging

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
from airside.detection.oakd.rerun_node import RerunNode
import time


ENABLE_RERUN = False
# If set to -1, the maximum difference theshold is ignored
MAX_TIMESTAMP_DIFF_SEC = 0.2


class OakD(AbstractCamera):
    def __init__(
        self,
        main_logger: logging.Logger,
        detections_logger: logging.Logger,
        detailed_detections_logger: logging.Logger,
        stop_event: threading.Event,
    ):
        super().__init__(
            main_logger, detections_logger, detailed_detections_logger, stop_event
        )

    def run(self):
        with dai.Pipeline() as pipeline:
            cameraBundle = CameraBundle(pipeline)
            # qDetections, qFrame = add_object_tracker(pipeline, cameraBundle)
            qDetections = add_object_tracker(pipeline, cameraBundle)
            add_basalt_vio_rtab(pipeline, cameraBundle)

            slam = cameraBundle.slam
            qSlamTransform = slam.transform.createOutputQueue(maxSize=1, blocking=False)
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

            try:
                while not self.stop_event.is_set():
                    time.sleep(0.01)

                    transform_msg = qSlamTransform.tryGet()
                    if isinstance(transform_msg, dai.TransformData):
                        quat = transform_msg.getQuaternion()
                        trans = transform_msg.getTranslation()
                        transform_timestamp = (
                            transform_msg.getTimestamp().total_seconds()
                        )

                    # Get tracker outputs
                    detectionMsg = qDetections.tryGet()
                    if detectionMsg and isinstance(
                        detectionMsg, dai.SpatialImgDetections
                    ):
                        if transform_timestamp is None or quat is None or trans is None:
                            self.main_logger.debug(
                                "Skipping detection: waiting for SLAM"
                            )
                            continue

                        detections_timestamp = (
                            detectionMsg.getTimestamp().total_seconds()
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
                            self.detailed_detections_logger.info(
                                f"Result: {mapped_target} | Pose: {origin_cam_coord} - {origin_cam_q} | Detection: {cam_target_coord}"
                            )
            finally:
                slam.saveDatabase()
                pipeline.stop()
        self.main_logger.info("Stopping OakD thread.")
