#!/usr/bin/env python3

from typing import Optional
import depthai as dai
from airside.detection.oakd.camera_bundle import CameraBundle


def add_object_tracker(
    p: dai.Pipeline,
    cameras: Optional["CameraBundle"] = None,
    use_default_model: bool = True,
    custom_model_name: str = "yolov6-nano",
):
    cameras = cameras or CameraBundle(p)
    depthNode = cameras.stereo
    depthNode.setExtendedDisparity(True)

    spatial_detection_node = p.create(dai.node.SpatialDetectionNetwork)
    if use_default_model:
        # Use the SDK's built-in default model selection path.
        build_fn = getattr(spatial_detection_node, "build")
        spatialDetectionNetwork = build_fn(cameras.camRgb, depthNode)
    else:
        spatialDetectionNetwork = spatial_detection_node.build(
            cameras.camRgb,
            depthNode,
            dai.NNModelDescription(custom_model_name),
        )

    spatialDetectionNetwork.setConfidenceThreshold(0.6)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    qDetections = spatialDetectionNetwork.out.createOutputQueue(
        maxSize=1, blocking=False
    )
    return qDetections
