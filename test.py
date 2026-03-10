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
            "RGBD/CreateOccupancyGrid": "false",
            "Grid/3D": "false",
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
    depth_node = pipeline.create(dai.node.StereoDepth)
    cameras.leftOutput.link(depth_node.left)
    cameras.rightOutput.link(depth_node.right)

    depth_node.setExtendedDisparity(True)
    depth_node.setOutputSize(640, 400)
    depth_node.setLeftRightCheck(True)
    depth_node.setSubpixel(True)
    depth_node.enableDistortionCorrection(False)
    depth_node.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    model_desc = dai.NNModelDescription("yolov6-nano")
    model_path_getter = getattr(model_desc, "getModelPath", None)

    using_explicit_imagealign = False

    if callable(model_path_getter):
        model_path = model_path_getter()
        if isinstance(model_path, (str, Path)):
            rgb_output = cameras.camRgb.requestOutput((640, 400))
            image_align = pipeline.create(dai.node.ImageAlign)
            image_align.setOutputSize(640, 400)
            depth_node.depth.link(image_align.input)
            rgb_output.link(image_align.inputAlignTo)

            # Preferred path: manual SDN so aligned depth is explicitly injected.
            spatial_detection_network = pipeline.create(dai.node.SpatialDetectionNetwork)
            spatial_detection_network.setBlobPath(Path(model_path))
            rgb_output.link(spatial_detection_network.input)
            image_align.outputAligned.link(spatial_detection_network.inputDepth)
            using_explicit_imagealign = True
        else:
            # Fallback for unexpected API behavior.
            spatial_detection_network = pipeline.create(
                dai.node.SpatialDetectionNetwork
            ).build(
                cameras.camRgb,
                depth_node,
                model_desc,
            )
    else:
        # Older/newer SDK bindings may not expose model path; keep pipeline runnable.
        spatial_detection_network = pipeline.create(dai.node.SpatialDetectionNetwork).build(
            cameras.camRgb,
            depth_node,
            model_desc,
        )
    spatial_detection_network.setConfidenceThreshold(0.6)
    spatial_detection_network.input.setBlocking(False)
    spatial_detection_network.setBoundingBoxScaleFactor(0.5)
    spatial_detection_network.setDepthLowerThreshold(100)
    spatial_detection_network.setDepthUpperThreshold(5000)

    if using_explicit_imagealign:
        print("[tracker] using explicit ImageAlign -> inputDepth path")
    else:
        print("[tracker] fallback build path (SDK did not expose model path API)")

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
                try:
                    transform_msg = q_slam_transform.tryGet()
                except (RuntimeError, Exception) as e:
                    print(f"[camera-test] transform queue closed: {e}")
                    break
                if isinstance(transform_msg, dai.TransformData):
                    transforms_seen += 1
                    trans = transform_msg.getTranslation()
                    quat = transform_msg.getQuaternion()
                    print(
                        "[transform] "
                        f"xyz=({trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f}) "
                        f"quat=({quat.qw:.4f}, {quat.qx:.4f}, {quat.qy:.4f}, {quat.qz:.4f})"
                    )

                try:
                    detections_msg = q_detections.tryGet()
                except (RuntimeError, Exception) as e:
                    print(f"[camera-test] detections queue closed: {e}")
                    break
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

                try:
                    obstacle_msg = q_obstacle_pcl.tryGet()
                except (RuntimeError, Exception):
                    obstacle_msg = None
                if obstacle_msg is not None:
                    print("[obstacle-pcl] received")

                try:
                    ground_msg = q_ground_pcl.tryGet()
                except (RuntimeError, Exception):
                    ground_msg = None
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
