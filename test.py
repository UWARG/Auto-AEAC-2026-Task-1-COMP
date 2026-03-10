#!/usr/bin/env python3

from pathlib import Path
import argparse
import time

import depthai as dai


class CameraBundle:
    """Helper to create and link camera nodes and stereo outputs."""

    def __init__(
        self,
        pipeline: dai.Pipeline,
        rgb_socket=dai.CameraBoardSocket.CAM_A,
        left_socket=dai.CameraBoardSocket.CAM_B,
        right_socket=dai.CameraBoardSocket.CAM_C,
        mono_resolution=(640, 400),
    ):
        self.pipeline = pipeline
        self.camRgb = pipeline.create(dai.node.Camera).build(rgb_socket)

        # This method exists on some DepthAI camera variants.
        set_preview_keep_aspect_ratio = getattr(
            self.camRgb, "setPreviewKeepAspectRatio", None
        )
        if callable(set_preview_keep_aspect_ratio):
            set_preview_keep_aspect_ratio(False)

        self.monoLeft = pipeline.create(dai.node.Camera).build(left_socket)
        self.monoRight = pipeline.create(dai.node.Camera).build(right_socket)
        self.slam = pipeline.create(dai.node.RTABMapSLAM)

        self.stereo = pipeline.create(dai.node.StereoDepth)
        self.leftOutput = self.monoLeft.requestOutput(mono_resolution)
        self.rightOutput = self.monoRight.requestOutput(mono_resolution)
        self.leftOutput.link(self.stereo.left)
        self.rightOutput.link(self.stereo.right)


def add_basalt_vio_rtab(pipeline: dai.Pipeline, cameras: CameraBundle) -> None:
    """Attach IMU + BasaltVIO + RTABMapSLAM graph using shared stereo."""
    output_folder = Path(__file__).parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    imu = pipeline.create(dai.node.IMU)
    odom = pipeline.create(dai.node.BasaltVIO)
    stereo = cameras.stereo
    slam = cameras.slam

    slam.setDatabasePath(str(output_folder / "building_test.db"))
    slam.setParams(
        {
            "RGBD/CreateOccupancyGrid": "true",
            "Grid/3D": "true",
            "Rtabmap/SaveWMState": "true",
        }
    )

    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200
    )
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setRectifyEdgeFillColor(0)
    stereo.enableDistortionCorrection(True)
    stereo.initialConfig.setLeftRightCheckThreshold(10)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

    stereo.syncedLeft.link(odom.left)
    stereo.syncedRight.link(odom.right)
    stereo.depth.link(slam.depth)
    stereo.rectifiedLeft.link(slam.rect)
    imu.out.link(odom.imu)
    odom.transform.link(slam.odom)


def add_object_tracker(pipeline: dai.Pipeline, cameras: CameraBundle):
    """Attach YOLO spatial detector and return detections queue."""
    depth_node = cameras.stereo
    depth_node.setExtendedDisparity(True)
    depth_node.setOutputSize(640, 400)

    spatial_detection_network = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        cameras.camRgb, depth_node, dai.NNModelDescription("yolov6-nano")
    )
    spatial_detection_network.setConfidenceThreshold(0.6)
    spatial_detection_network.input.setBlocking(False)
    spatial_detection_network.setBoundingBoxScaleFactor(0.5)
    spatial_detection_network.setDepthLowerThreshold(100)
    spatial_detection_network.setDepthUpperThreshold(5000)

    return spatial_detection_network.out.createOutputQueue(maxSize=16, blocking=False)


def run_camera_test(duration_sec: int) -> None:
    """Run the camera pipeline for a fixed time and print node outputs."""
    with dai.Pipeline() as pipeline:
        cameras = CameraBundle(pipeline)
        q_detections = add_object_tracker(pipeline, cameras)
        add_basalt_vio_rtab(pipeline, cameras)

        slam = cameras.slam
        q_slam_transform = slam.transform.createOutputQueue(maxSize=1, blocking=False)
        q_obstacle_pcl = slam.obstaclePCL.createOutputQueue(maxSize=1, blocking=False)
        q_ground_pcl = slam.groundPCL.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        print("[camera-test] Pipeline started")

        start = time.time()
        last_report = 0.0
        detections_seen = 0
        transforms_seen = 0

        try:
            while time.time() - start < duration_sec:
                transform_msg = q_slam_transform.tryGet()
                if isinstance(transform_msg, dai.TransformData):
                    transforms_seen += 1
                    trans = transform_msg.getTranslation()
                    quat = transform_msg.getQuaternion()
                    print(
                        "[transform] "
                        f"xyz=({trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f}) "
                        f"quat=({quat.qw:.4f}, {quat.qx:.4f}, {quat.qy:.4f}, {quat.qz:.4f})"
                    )

                detections_msg = q_detections.tryGet()
                if detections_msg and isinstance(
                    detections_msg, dai.SpatialImgDetections
                ):
                    num = len(detections_msg.detections)
                    detections_seen += num
                    print(f"[detections] count={num}")
                    for det in detections_msg.detections:
                        xyz = det.spatialCoordinates
                        print(
                            "  - "
                            f"label={det.label} conf={det.confidence:.3f} "
                            f"xyz_mm=({xyz.x:.1f}, {xyz.y:.1f}, {xyz.z:.1f})"
                        )

                obstacle_msg = q_obstacle_pcl.tryGet()
                if obstacle_msg is not None:
                    print("[obstacle-pcl] received")

                ground_msg = q_ground_pcl.tryGet()
                if ground_msg is not None:
                    print("[ground-pcl] received")

                now = time.time()
                if now - last_report >= 1.0:
                    elapsed = int(now - start)
                    print(
                        f"[status t={elapsed}s] transforms={transforms_seen} detections={detections_seen}"
                    )
                    last_report = now

                time.sleep(0.01)
        finally:
            slam.saveDatabase()
            pipeline.stop()
            print("[camera-test] Pipeline stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone OAK-D camera pipeline test"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="How long to run the camera test in seconds (default: 20)",
    )
    args = parser.parse_args()

    run_camera_test(duration_sec=args.duration)
