#!/usr/bin/env python3

from typing import Optional
import depthai as dai
from airside.detection.oakd.camera_bundle import CameraBundle


def add_object_tracker(p: dai.Pipeline, cameras: Optional["CameraBundle"] = None):
    cameras = cameras or CameraBundle(p)
    depthNode = cameras.stereo
    depthNode.setExtendedDisparity(True)

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        cameras.camRgb, depthNode, dai.NNModelDescription("yolov6-nano")
    )

    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    qDetections = spatialDetectionNetwork.out.createOutputQueue(
        maxSize=1, blocking=False
    )
    return qDetections
