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
    depthNode.enableDistortionCorrection(False)
    depthNode.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    imageAlign = p.create(dai.node.ImageAlign)
    imageAlign.setOutputSize(640, 400)
    depthNode.depth.link(imageAlign.input)
    cameras.camRgb.requestOutput((640, 400)).link(imageAlign.inputAlignTo)

    # Create spatial detection network with aligned depth
    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork)
    spatialDetectionNetwork.setBlob(
        dai.OpenVINO.Blob(dai.OpenVINO.Version.VERSION_2021_4, "yolov6n_coco_640x640")
    )
    spatialDetectionNetwork.setNumInferenceThreads(2)

    # Connect RGB for detection and aligned depth for spatial calculations
    cameras.camRgb.requestOutput((640, 400)).link(spatialDetectionNetwork.input)
    imageAlign.outputAligned.link(spatialDetectionNetwork.inputDepth)

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
