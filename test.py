#!/usr/bin/env python3

import depthai as dai
import time

# ----------------------------
# Pipeline
# ----------------------------

pipeline = dai.Pipeline()

# ----------------------------
# Cameras (Software v3 API)
# ----------------------------

# RGB
rgb = pipeline.create(dai.node.Camera).build()
rgbPreview = rgb.requestOutput((640, 640), type=dai.ImgFrame.Type.RGB888p)

# Left mono
left = pipeline.create(dai.node.Camera).build()
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
leftOut = left.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

# Right mono
right = pipeline.create(dai.node.Camera).build()
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
rightOut = right.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

# ----------------------------
# Stereo Depth
# ----------------------------

stereo = pipeline.create(dai.node.StereoDepth)

leftOut.link(stereo.left)
rightOut.link(stereo.right)

depthOut = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

# ----------------------------
# Spatial Detection Network
# ----------------------------

spatialNN = pipeline.create(dai.node.SpatialDetectionNetwork)

spatialNN.setModel(dai.NNModelDescription("yolov6-nano"))
spatialNN.setConfidenceThreshold(0.5)

rgbPreview.link(spatialNN.input)
stereo.depth.link(spatialNN.inputDepth)

detectionsQueue = spatialNN.out.createOutputQueue(maxSize=1, blocking=False)

# ----------------------------
# IMU
# ----------------------------

imu = pipeline.create(dai.node.IMU)

imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

# ----------------------------
# VIO
# ----------------------------

vio = pipeline.create(dai.node.Vio)

stereo.rectifiedLeft.link(vio.left)
stereo.rectifiedRight.link(vio.right)
imu.out.link(vio.imu)

vioQueue = vio.out.createOutputQueue(maxSize=1, blocking=False)

# ----------------------------
# Start Pipeline
# ----------------------------

pipeline.start()

print("Pipeline started")

# ----------------------------
# Main loop
# ----------------------------

while pipeline.isRunning():

    # -----------------------
    # Spatial detections
    # -----------------------
    det = detectionsQueue.tryGet()

    if det is not None:
        for d in det.detections:
            xyz = d.spatialCoordinates
            print(
                f"DET label={d.label} conf={d.confidence:.2f} "
                f"x={xyz.x:.0f} y={xyz.y:.0f} z={xyz.z:.0f} mm"
            )

    # -----------------------
    # VIO pose
    # -----------------------
    vioData = vioQueue.tryGet()

    if vioData is not None:
        p = vioData.pose.position
        q = vioData.pose.rotation

        print(
            f"VIO pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) "
            f"quat=({q.i:.3f},{q.j:.3f},{q.k:.3f},{q.real:.3f})"
        )

    time.sleep(0.005)
