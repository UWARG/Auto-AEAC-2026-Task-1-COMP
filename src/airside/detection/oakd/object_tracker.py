#!/usr/bin/env python3

from typing import Optional
import depthai as dai
from airside.detection.oakd.camera_bundle import CameraBundle


def add_object_tracker(p: dai.Pipeline, cameras: Optional["CameraBundle"] = None):
    cameras = cameras or CameraBundle(p)
    depthNode = p.create(dai.node.StereoDepth)
    cameras.leftOutput.link(depthNode.left)
    cameras.rightOutput.link(depthNode.right)

    depthNode.setExtendedDisparity(True)
    depthNode.setOutputSize(640, 400)
    depthNode.setLeftRightCheck(True)
    depthNode.setSubpixel(True)
    depthNode.enableDistortionCorrection(True)
    depthNode.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        cameras.camRgb, depthNode, dai.NNModelDescription("yolov6-nano")
    )

    spatialDetectionNetwork.setConfidenceThreshold(0.6)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    qDetections = spatialDetectionNetwork.out.createOutputQueue(
        maxSize=1, blocking=False
    )
    # qFrame = spatialDetectionNetwork.passthrough.createOutputQueue(maxSize=16, blocking=False)
    # return qDetections, qFrame
    return qDetections
