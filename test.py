#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import depthai as dai
import numpy as np

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--depthSource", type=str, default="stereo", choices=["stereo", "neural"]
)
args = parser.parse_args()
# For better results on OAK4, use a segmentation model like "luxonis/yolov8-instance-segmentation-large:coco-640x480"
# for depth estimation over the objects mask instead of the full bounding box.
modelDescription = dai.NNModelDescription("yolov6-nano")
size = (640, 400)

if args.depthSource == "stereo":
    fps = STEREO_DEFAULT_FPS
else:
    fps = NEURAL_FPS

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        depthPreview = depthPreview.getCvFrame()
        rgbPreview = rgbPreview.getCvFrame()
        depthFrameColor = self.processDepthFrame(depthPreview)
        self.displayResults(rgbPreview, depthFrameColor, detections.detections)

    def processDepthFrame(self, depthFrame):
        depthDownscaled = depthFrame[::4]
        if np.all(depthDownscaled == 0):
            minDepth = 0
        else:
            minDepth = np.percentile(depthDownscaled[depthDownscaled != 0], 1)
        maxDepth = np.percentile(depthDownscaled, 99)
        depthFrameColor = np.interp(depthFrame, (minDepth, maxDepth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    def displayResults(self, rgbFrame, depthFrameColor, detections):
        height, width, _ = rgbFrame.shape
        for detection in detections:
            self.drawBoundingBoxes(depthFrameColor, detection)
            self.drawDetections(rgbFrame, detection, width, height)

        cv2.imshow("Depth frame", depthFrameColor)
        cv2.imshow("Color frame", rgbFrame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def drawBoundingBoxes(self, depthFrameColor, detection):
        roiData = detection.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        cv2.rectangle(depthFrameColor, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), (255, 255, 255), 1)

    def drawDetections(self, frame, detection, frameWidth, frameHeight):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        color = (255, 255, 255)
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

# Creates the pipeline and a default device implicitly
with dai.Pipeline() as p:
    # Define sources and outputs
    platform = p.getDefaultDevice().getPlatform()

    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    if args.depthSource == "stereo":
        depthSource = p.create(dai.node.StereoDepth)
        depthSource.setExtendedDisparity(True)
        monoLeft.requestOutput(size).link(depthSource.left)
        monoRight.requestOutput(size).link(depthSource.right)
    elif args.depthSource == "neural":
        depthSource = p.create(dai.node.NeuralDepth).build(
            monoLeft.requestFullResolutionOutput(),
            monoRight.requestFullResolutionOutput(),
            dai.DeviceModelZoo.NEURAL_DEPTH_LARGE,
        )
    else:
        raise ValueError(f"Invalid depth source: {args.depthSource}")

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        camRgb, depthSource, modelDescription
    )
    visualizer = p.create(SpatialVisualizer)

    spatialDetectionNetwork.spatialLocationCalculator.initialConfig.setSegmentationPassthrough(False)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    visualizer.build(
        spatialDetectionNetwork.passthroughDepth,
        spatialDetectionNetwork.out,
        spatialDetectionNetwork.passthrough,
    )

    print("Starting pipeline with depth source: ", args.depthSource)

    p.run()


# #!/usr/bin/env python3

# import time
# import depthai as dai

# NEURAL_FPS = 8
# STEREO_DEFAULT_FPS = 20

# fps = STEREO_DEFAULT_FPS
# size = (640, 400)

# modelDescription = dai.NNModelDescription("yolov6-nano")

# with dai.Pipeline() as p:

#     # -----------------------------
#     # Cameras
#     # -----------------------------

#     camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
#     monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
#     monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)

#     # -----------------------------
#     # Stereo depth
#     # -----------------------------

#     stereo = p.create(dai.node.StereoDepth)

#     stereo.setExtendedDisparity(True)

#     monoLeft.requestOutput(size).link(stereo.left)
#     monoRight.requestOutput(size).link(stereo.right)

#     # -----------------------------
#     # Spatial detection
#     # -----------------------------

#     spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
#         camRgb,
#         stereo,
#         modelDescription,
#     )

#     spatialDetectionNetwork.input.setBlocking(False)
#     spatialDetectionNetwork.setDepthLowerThreshold(100)
#     spatialDetectionNetwork.setDepthUpperThreshold(5000)

#     # -----------------------------
#     # IMU
#     # -----------------------------

#     imu = p.create(dai.node.IMU)

#     imu.enableIMUSensor(
#         [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
#         200,
#     )

#     imu.setBatchReportThreshold(1)
#     imu.setMaxBatchReports(10)

#     # -----------------------------
#     # VIO / SLAM
#     # -----------------------------

#     odom = p.create(dai.node.BasaltVIO)
#     slam = p.create(dai.node.RTABMapSLAM)

#     params = {
#         "RGBD/CreateOccupancyGrid": "true",
#         "Grid/3D": "true",
#         "Rtabmap/SaveWMState": "true",
#     }

#     slam.setParams(params)

#     stereo.syncedLeft.link(odom.left)
#     stereo.syncedRight.link(odom.right)
#     imu.out.link(odom.imu)

#     stereo.depth.link(slam.depth)
#     stereo.rectifiedLeft.link(slam.rect)
#     odom.transform.link(slam.odom)

#     # -----------------------------
#     # Output queues
#     # -----------------------------

#     detQ = spatialDetectionNetwork.out.createOutputQueue(maxSize=1, blocking=False)
#     slamQ = slam.transform.createOutputQueue(maxSize=1, blocking=False)

#     print("Starting pipeline")

#     p.start()

#     while p.isRunning():

#         # -----------------------------
#         # Detections
#         # -----------------------------

#         det = detQ.tryGet()

#         if det is not None:
#             print("\nDetections:")

#             for detection in det.detections:

#                 coords = detection.spatialCoordinates

#                 print(
#                     f"label={detection.labelName} "
#                     f"conf={detection.confidence:.2f} "
#                     f"x={int(coords.x)}mm "
#                     f"y={int(coords.y)}mm "
#                     f"z={int(coords.z)}mm"
#                 )

#         # -----------------------------
#         # SLAM pose
#         # -----------------------------

#         slamData = slamQ.tryGet()

#         if slamData is not None:

#             pos = slamData.getTranslation()
#             rot = slamData.getQuaternion()

#             print("\nSLAM Pose:")

#             print(
#                 f"pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) "
#                 f"quat=({rot.qx:.3f}, {rot.qy:.3f}, {rot.qz:.3f}, {rot.qw:.3f})"
#             )

#         time.sleep(0.01)
