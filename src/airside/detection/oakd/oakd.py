import threading
import logging

from airside.detection import FRD_conversion
from airside.detection.oakd.camera_bundle import CameraBundle

from ..abstract_camera import AbstractCamera
from util import Colours, MappedTarget, Direction, Coordinate, Colour, Quaternion, Target
from airside.detection.oakd.Basalt_VIO_RTab import add_basalt_vio_rtab
from airside.detection.oakd.object_tracker import add_object_tracker
import depthai as dai
from airside.detection.oakd.rerun_node import RerunNode
import time
from airside.detection.oakd.camera_bundle import CameraBundle


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
            qTracklets, qFrame = add_object_tracker(pipeline, cameraBundle)
            add_basalt_vio_rtab(pipeline, cameraBundle)

            slam = cameraBundle.slam
            if ENABLE_RERUN:
                rerunViewer = RerunNode()
                slam.transform.link(rerunViewer.inputTrans)
                slam.passthroughRect.link(rerunViewer.inputImg)
                slam.occupancyGridMap.link(rerunViewer.inputGrid)
                slam.obstaclePCL.link(rerunViewer.inputObstaclePCL)
                slam.groundPCL.link(rerunViewer.inputGroundPCL)

            pipeline.start()
            self.main_logger.info("Starting Pipeline...")
            try:
                while not self.stop_event.is_set():
                    # Get tracker outputs
                    trackMsg = qTracklets.tryGet()
                    frameMsg = qFrame.tryGet()
                    # print tracked targets only when being tracked    
                    if trackMsg:
                        self.main_logger.info(f"Track Message has item: {trackMsg}")
                        for t in trackMsg.tracklets: # type: ignore
                            # Log only when actively tracked to minimize spam
                            if t.status.name in ("TRACKED", "NEW"):
                                sc = getattr(t, "spatialCoordinates", None)
                                if sc:
                                    transform = cameraBundle.slam.getLocalTransform()
                                    quat = transform.getQuaternion()
                                    trans = transform.getTranslation()
                                    
                                    translated_coordinate = FRD_conversion.convert_target_to_FRD(
                                        cam_target_coord=Coordinate(sc.x / 1000.0, sc.y / 1000.0, sc.z / 1000.0),
                                        origin_cam_q=Quaternion(quat.qw, quat.qx, quat.qy, quat.qz),
                                        origin_cam_coord=Coordinate(trans.x, trans.y, trans.z),
                                    )

                                    mapped_target = Target(
                                        colour=Colours.RED,  # TODO: Replace with actual colour
                                        location=translated_coordinate
                                    )

                                    self.detections_logger.info(mapped_target)
                                    self.main_logger.info(f"Detected target: {mapped_target}")
                    time.sleep(0.01)
            finally:
                slam.saveDatabase()
                pipeline.stop()
        self.main_logger.info("Stopping OakD thread.")
