import depthai as dai
import time


pipeline = dai.Pipeline()


colorCam = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)

colorCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setPreviewSize(640, 640)
colorCam.setInterleaved(False)

left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)


stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

left.out.link(stereo.left)
right.out.link(stereo.right)


spatialDetection = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

spatialDetection.setBlobPath(
    dai.OpenVINO.Blob(
        dai.OpenVINO.Version.VERSION_2021_4,
        dai.NNModelDescription(
            "depthai/yolov6-nano"
        )
    )
)

spatialDetection.setConfidenceThreshold(0.5)
spatialDetection.setNumClasses(80)
spatialDetection.setCoordinateSize(4)
spatialDetection.setAnchors([])
spatialDetection.setAnchorMasks({})
spatialDetection.setIouThreshold(0.5)

colorCam.preview.link(spatialDetection.input)
stereo.depth.link(spatialDetection.inputDepth)


imu = pipeline.create(dai.node.IMU)
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)


vio = pipeline.create(dai.node.Vio)

stereo.rectifiedLeft.link(vio.left)
stereo.rectifiedRight.link(vio.right)
imu.out.link(vio.imu)


xoutDet = pipeline.create(dai.node.XLinkOut)
xoutDet.setStreamName("detections")

xoutVio = pipeline.create(dai.node.XLinkOut)
xoutVio.setStreamName("vio")

spatialDetection.out.link(xoutDet.input)
vio.out.link(xoutVio.input)


with dai.Device(pipeline) as device:

    detQueue = device.getOutputQueue("detections", maxSize=1, blocking=False)
    vioQueue = device.getOutputQueue("vio", maxSize=1, blocking=False)

    print("Pipeline started")

    while True:

        
        det = detQueue.tryGet()

        if det is not None:
            detections = det.detections
            print("\nDetections:")

            for d in detections:
                print(
                    f"Label: {d.label} "
                    f"Conf: {d.confidence:.2f} "
                    f"XYZ: ({d.spatialCoordinates.x:.1f}, "
                    f"{d.spatialCoordinates.y:.1f}, "
                    f"{d.spatialCoordinates.z:.1f}) mm"
                )

        
        vioData = vioQueue.tryGet()

        if vioData is not None:
            pose = vioData.pose

            pos = pose.position
            rot = pose.rotation

            print("\nVIO Pose:")
            print(
                f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) m"
            )
            print(
                f"Quaternion: ({rot.i:.3f}, {rot.j:.3f}, {rot.k:.3f}, {rot.real:.3f})"
            )

        time.sleep(0.01)