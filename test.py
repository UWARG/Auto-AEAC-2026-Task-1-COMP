#!/usr/bin/env python3

import depthai as dai
import time

pipeline = dai.Pipeline()

# --------------------------------------------------
# Cameras (Software v3)
# --------------------------------------------------

# RGB camera
rgbCam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
rgbPreview = rgbCam.requestOutput((640, 640), type=dai.ImgFrame.Type.RGB888p)

# Left mono
leftCam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
leftOut = leftCam.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

# Right mono
rightCam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
rightOut = rightCam.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

# --------------------------------------------------
# Stereo Depth
# --------------------------------------------------

stereo = pipeline.create(dai.node.StereoDepth)

leftOut.link(stereo.left)
rightOut.link(stereo.right)

# --------------------------------------------------
# Spatial Detection
# --------------------------------------------------

spatialNN = pipeline.create(dai.node.SpatialDetectionNetwork)

spatialNN.setModel(dai.NNModelDescription("yolov6-nano"))
spatialNN.setConfidenceThreshold(0.5)

rgbPreview.link(spatialNN.input)
stereo.depth.link(spatialNN.inputDepth)

detectionsQueue = spatialNN.out.createOutputQueue(
    maxSize=1,
    blocking=False
)

# --------------------------------------------------
# IMU
# --------------------------------------------------

imu = pipeline.create(dai.node.IMU)

imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

# --------------------------------------------------
# VIO
# --------------------------------------------------

vio = pipeline.create(dai.node.Vio)

stereo.rectifiedLeft.link(vio.left)
stereo.rectifiedRight.link(vio.right)
imu.out.link(vio.imu)

vioQueue = vio.out.createOutputQueue(
    maxSize=1,
    blocking=False
)

# --------------------------------------------------
# Start pipeline
# --------------------------------------------------

pipeline.start()

print("Pipeline running")

# --------------------------------------------------
# Main loop
# --------------------------------------------------

while pipeline.isRunning():

    det = detectionsQueue.tryGet()
    if det:
        for d in det.detections:
            xyz = d.spatialCoordinates
            print(
                f"DET label={d.label} "
                f"conf={d.confidence:.2f} "
                f"x={xyz.x:.0f} y={xyz.y:.0f} z={xyz.z:.0f} mm"
            )

    vioData = vioQueue.tryGet()
    if vioData:
        pos = vioData.pose.position
        rot = vioData.pose.rotation

        print(
            f"VIO pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) "
            f"quat=({rot.i:.3f},{rot.j:.3f},{rot.k:.3f},{rot.real:.3f})"
        )

    time.sleep(0.005)
