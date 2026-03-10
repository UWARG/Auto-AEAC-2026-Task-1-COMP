#!/usr/bin/env python3

"""Standalone OAK-D test pipeline for spatial detections + VIO/SLAM.

This script builds a single DepthAI pipeline with:
- Spatial detections using default YOLO model (yolov6-nano)
- Basalt VIO feeding RTABMap SLAM

Both key output queues are configured with maxSize=1 and non-blocking mode:
- Spatial detections queue
- SLAM transform queue
"""

from __future__ import annotations

from typing import Any

import depthai as dai

from src.airside.detection.oakd.camera_bundle import CameraBundle
from src.airside.detection.oakd.Basalt_VIO_RTab import add_basalt_vio_rtab


def build_pipeline() -> tuple[dai.Pipeline, Any, Any]:
    """Create and start a pipeline, returning detection and transform queues."""
    pipeline = dai.Pipeline()

    with pipeline:
        cameras = CameraBundle(pipeline)

        # Spatial YOLO detection on RGB + stereo depth.
        depth_node = cameras.stereo
        depth_node.setExtendedDisparity(True)
        depth_node.setOutputSize(640, 400)

        spatial_detection_network = pipeline.create(
            dai.node.SpatialDetectionNetwork
        ).build(cameras.camRgb, depth_node, dai.NNModelDescription("yolov6-nano"))
        spatial_detection_network.setConfidenceThreshold(0.6)
        spatial_detection_network.input.setBlocking(False)
        spatial_detection_network.setBoundingBoxScaleFactor(0.5)
        spatial_detection_network.setDepthLowerThreshold(100)
        spatial_detection_network.setDepthUpperThreshold(5000)

        # Add Basalt VIO + RTABMap SLAM onto the same camera bundle.
        add_basalt_vio_rtab(pipeline, cameras)

        q_detections = spatial_detection_network.out.createOutputQueue(
            maxSize=1,
            blocking=False,
        )
        q_transform = cameras.slam.transform.createOutputQueue(
            maxSize=1,
            blocking=False,
        )

    pipeline.start()
    return pipeline, q_detections, q_transform


def main() -> None:
    pipeline, q_detections, q_transform = build_pipeline()

    print("Pipeline started: YOLO spatial detections + Basalt/RTABMap VIO/SLAM")
    print("Queues configured with maxSize=1 (detections + transform)")

    try:
        while True:
            det_msg = q_detections.tryGet()
            tf_msg = q_transform.tryGet()

            if isinstance(det_msg, dai.SpatialImgDetections):
                print(f"detections={len(det_msg.detections)}")

            if isinstance(tf_msg, dai.TransformData):
                translation = tf_msg.getTranslation()
                print(
                    "transform="
                    f"({translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f})"
                )

    except KeyboardInterrupt:
        print("Stopping pipeline...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
