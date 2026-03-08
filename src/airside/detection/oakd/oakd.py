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
 
 
class OakD(AbstractCamera):
    def __init__(
        self,
        main_logger: logging.Logger,
        detections_logger: logging.Logger,
        stop_event: threading.Event,
    ):
        super().__init__(main_logger, detections_logger, stop_event)
 
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
            latest_transform: dai.TransformData | None = None
            try:
                while not self.stop_event.is_set():
                    transform_msg = qSlamTransform.tryGet()
                    if transform_msg is not None:
                        latest_transform = transform_msg  # type: ignore
                    else:
                        continue
                    quat = latest_transform.getQuaternion() # type: ignore
                    trans = latest_transform.getTranslation() # type: ignore
 
                    qw, qx, qy, qz = (
                        quat.qw,
                        quat.qx,
                        quat.qy,
                        quat.qz,
                    )
 
                    # Get tracker outputs
                    detectionMsg = qDetections.tryGet()
                    if detectionMsg:
                        for detection in detectionMsg.detections:  # type: ignore
                            detection: dai.SpatialImgDetection = detection
                            sc = getattr(detection, "spatialCoordinates", None)
                            if sc is None:
                                self.main_logger.error(
                                    "Detection missing spatial coordinates"
                                )
                                continue
                            if sc:
                                if latest_transform is None:
                                    self.main_logger.debug(
                                        "Skipping detection: waiting for SLAM"
                                    )
                                    continue
 
                            cam_target_coord = Coordinate(
                                sc.z / 1000.0, sc.x / 1000.0, sc.y / 1000.0
                            )
                            origin_cam_q = Quaternion(qw, qx, -qy, -qz)
                            origin_cam_coord = Coordinate(
                                trans.x, -trans.y, -trans.z
                            )
 
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
                            
                            # self.main_logger.info(
                            #     f"Detected target: {translated_coordinate}"
                            # )
                            # self.main_logger.info(
                            #     f"Origin-Cam Coord: {origin_cam_coord}"
                            # )
                            # self.main_logger.info(
                            #     f"Origin-Cam Quat: {origin_cam_q}"
                            # )
                            # self.main_logger.info(
                            #     f"Cam-Target Coord: {cam_target_coord}"
                            # )
                            
                    time.sleep(0.01)
            finally:
                slam.saveDatabase()
                pipeline.stop()
        self.main_logger.info("Stopping OakD thread.")
