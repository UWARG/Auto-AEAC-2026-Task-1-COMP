#!/usr/bin/env python3

import time
import depthai as dai

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 20

fps = STEREO_DEFAULT_FPS
size = (640, 400)

modelDescription = dai.NNModelDescription("yolov6-nano")

with dai.Pipeline() as p:

    # -----------------------------
    # Cameras
    # -----------------------------

    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)

    # -----------------------------
    # Stereo depth
    # -----------------------------

    stereo = p.create(dai.node.StereoDepth)

    stereo.setExtendedDisparity(True)

    monoLeft.requestOutput(size).link(stereo.left)
    monoRight.requestOutput(size).link(stereo.right)

    # -----------------------------
    # Spatial detection
    # -----------------------------

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        camRgb,
        stereo,
        modelDescription,
    )

    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # -----------------------------
    # IMU
    # -----------------------------

    imu = p.create(dai.node.IMU)

    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
        200,
    )

    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # -----------------------------
    # VIO / SLAM
    # -----------------------------

    odom = p.create(dai.node.BasaltVIO)
    slam = p.create(dai.node.RTABMapSLAM)

    params = {
        "RGBD/CreateOccupancyGrid": "true",
        "Grid/3D": "true",
        "Rtabmap/SaveWMState": "true",
    }

    slam.setParams(params)

    stereo.syncedLeft.link(odom.left)
    stereo.syncedRight.link(odom.right)
    imu.out.link(odom.imu)

    stereo.depth.link(slam.depth)
    stereo.rectifiedLeft.link(slam.rect)
    odom.transform.link(slam.odom)

    # -----------------------------
    # Output queues
    # -----------------------------

    detQ = spatialDetectionNetwork.out.createOutputQueue(maxSize=1, blocking=False)
    slamQ = slam.transform.createOutputQueue(maxSize=1, blocking=False)

    print("Starting pipeline")

    p.start()

    while p.isRunning():

        # -----------------------------
        # Detections
        # -----------------------------

        det = detQ.tryGet()

        if det is not None:
            print("\nDetections:")

            for detection in det.detections:

                coords = detection.spatialCoordinates

                print(
                    f"label={detection.labelName} "
                    f"conf={detection.confidence:.2f} "
                    f"x={int(coords.x)}mm "
                    f"y={int(coords.y)}mm "
                    f"z={int(coords.z)}mm"
                )

        # -----------------------------
        # SLAM pose
        # -----------------------------

        slamData = slamQ.tryGet()

        if slamData is not None:

            pose = slamData.pose
            pos = pose.position
            rot = pose.orientation

            print("\nSLAM Pose:")

            print(
                f"pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) "
                f"quat=({rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f}, {rot.w:.3f})"
            )

        time.sleep(0.01)
