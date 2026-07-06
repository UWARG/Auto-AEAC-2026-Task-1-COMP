"""Demo of the airside OAK-D pipeline."""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

installExamplesStr = (
    Path(__file__).absolute().parents[2] / "install_requirements.py --install_rerun"
)
try:
    import rerun as rr
except ImportError:
    sys.exit(
        "Critical dependency missing: Rerun. Please install it using the command: '{} {}' and then rerun the script.".format(
            sys.executable, installExamplesStr
        )
    )

FPS = 60
WIDTH = 640
HEIGHT = 400

# Cap red-circle detection rate so the host-side OpenCV work does not starve
# RTAB-Map's SLAM backend (which also runs on the host CPU) of cycles.
DETECT_FPS_CAP = 15

# Depth gating for back-projected circles (millimetres).
DEPTH_LOWER_MM = 100
DEPTH_UPPER_MM = 15000

# Red-circle detection tuning.
MIN_CONTOUR_AREA = 200
MIN_CIRCULARITY = 0.6


class CircleSlamViewer(dai.node.ThreadedHostNode):
    """Visualizes SLAM output and red-circle detections together in Rerun."""

    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.inputTrans = dai.Node.Input(self)
        self.inputImg = dai.Node.Input(self)
        self.inputObstaclePCL = dai.Node.Input(self)
        self.inputGroundPCL = dai.Node.Input(self)
        self.inputGrid = dai.Node.Input(self)
        self.inputColorImg = dai.Node.Input(self)
        self.inputDepth = dai.Node.Input(self)
        # All queues drop stale frames rather than buffering them
        for inp in [self.inputTrans, self.inputImg, self.inputObstaclePCL,
                    self.inputGroundPCL, self.inputGrid, self.inputColorImg, self.inputDepth]:
            inp.setBlocking(False)
            inp.setMaxSize(1)
        self.positions = []
        self.slam_fx = 400.0
        self.slam_fy = 400.0
        self.det_fx = 400.0
        self.det_fy = 400.0
        self.det_cx = 320.0
        self.det_cy = 240.0
        self.slamIntrinsicsSet = False
        self.detIntrinsicsSet = False

    def _read_intrinsics(self, socket_int, width, height):
        cal = self.getParentPipeline().getDefaultDevice().readCalibration()
        m = cal.getCameraIntrinsics(dai.CameraBoardSocket(socket_int), width, height)
        return m[0][0], m[1][1], m[0][2], m[1][2]  # fx, fy, cx, cy

    @staticmethod
    def _quat_rotate(qx, qy, qz, qw, v):
        q = np.array([qx, qy, qz])
        t = 2.0 * np.cross(q, v)
        return v + qw * t + np.cross(q, t)

    @staticmethod
    def _detect_red_circles(bgr):
        """Return list of (px, py) pixel centres of red circular blobs."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   100, 80]), np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centres = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter ** 2) < MIN_CIRCULARITY:  # circularity
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centres.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        return centres

    def run(self):
        rr.init("", spawn=True)
        rr.log("world", rr.ViewCoordinates.FLU)
        rr.log("world/ground", rr.Boxes3D(half_sizes=[3.0, 3.0, 0.00001]))

        cached_position: rr.datatypes.Vec3D | None = None
        cached_quat_xyzw: tuple[float, float, float, float] | None = None
        cached_t_world: np.ndarray | None = None

        detect_interval = 1.0 / DETECT_FPS_CAP
        last_detect_time = 0.0

        while self.isRunning():
            # Drive the loop at camera rate, not SLAM rate
            colorFrame = self.inputColorImg.get()
            depthFrame = self.inputDepth.tryGet()
            transData  = self.inputTrans.tryGet()
            imgFrame   = self.inputImg.tryGet()

            # Update cached SLAM state whenever new data arrives
            if isinstance(transData, dai.TransformData):
                t = transData.getTranslation()
                q = transData.getQuaternion()
                cached_position  = rr.datatypes.Vec3D([t.x, t.y, t.z])
                cached_quat_xyzw = (q.qx, q.qy, q.qz, q.qw)
                cached_t_world   = np.array([t.x, t.y, t.z])

            # SLAM visualisation when both transform and image are fresh
            if isinstance(transData, dai.TransformData) and isinstance(imgFrame, dai.ImgFrame):
                if not self.slamIntrinsicsSet:
                    self.slam_fx, self.slam_fy, _, _ = self._read_intrinsics(
                        imgFrame.getInstanceNum(), imgFrame.getWidth(), imgFrame.getHeight()
                    )
                    self.slamIntrinsicsSet = True

                assert cached_position is not None and cached_quat_xyzw is not None
                rr.log("world/camera", rr.Transform3D(
                    translation=cached_position,
                    rotation=rr.datatypes.Quaternion(xyzw=list(cached_quat_xyzw)),
                ))
                self.positions.append(cached_position)
                rr.log("world/trajectory", rr.LineStrips3D(rr.components.LineStrip3D(self.positions)))
                rr.log("world/camera/image", rr.Pinhole(
                    resolution=[imgFrame.getWidth(), imgFrame.getHeight()],
                    focal_length=[self.slam_fx, self.slam_fy],
                    camera_xyz=rr.ViewCoordinates.FLU,
                ))
                img = imgFrame.getCvFrame()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rr.log("world/camera/image/rgb", rr.Image(img))

                pclObstData = self.inputObstaclePCL.tryGet()
                pclGrndData = self.inputGroundPCL.tryGet()
                mapData     = self.inputGrid.tryGet()
                if isinstance(pclObstData, dai.PointCloudData):
                    points, colors = pclObstData.getPointsRGB()
                    rr.log("world/obstacle_pcl", rr.Points3D(points, colors=colors, radii=[0.01]))
                if isinstance(pclGrndData, dai.PointCloudData):
                    points, colors = pclGrndData.getPointsRGB()
                    rr.log("world/ground_pcl", rr.Points3D(points, colors=colors, radii=[0.01]))
                if isinstance(mapData, dai.MapData):
                    rr.log("map", rr.Image(mapData.map.getCvFrame()))

            # Capps red circle detection
            now = time.monotonic()
            if now - last_detect_time < detect_interval:
                continue
            last_detect_time = now

            if cached_t_world is None or cached_quat_xyzw is None:
                continue
            if not isinstance(colorFrame, dai.ImgFrame) or not isinstance(depthFrame, dai.ImgFrame):
                continue

            if not self.detIntrinsicsSet:
                self.det_fx, self.det_fy, self.det_cx, self.det_cy = self._read_intrinsics(
                    colorFrame.getInstanceNum(), colorFrame.getWidth(), colorFrame.getHeight()
                )
                self.detIntrinsicsSet = True

            bgr      = colorFrame.getCvFrame()
            depth_mm = depthFrame.getFrame().astype(np.float32)

            if depth_mm.shape[:2] != bgr.shape[:2]:
                depth_mm = cv2.resize(depth_mm, (bgr.shape[1], bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            qx, qy, qz, qw = cached_quat_xyzw
            world_positions = []

            for (px, py) in self._detect_red_circles(bgr):
                z_mm = float(depth_mm[py, px])
                if z_mm < DEPTH_LOWER_MM or z_mm > DEPTH_UPPER_MM:
                    continue
                x_mm = (px - self.det_cx) * z_mm / self.det_fx
                y_mm = (py - self.det_cy) * z_mm / self.det_fy
                # Convert RDF (right, down, forward) → FLU (forward, left, up)
                cam_flu = np.array([z_mm / 1000.0, -x_mm / 1000.0, -y_mm / 1000.0])
                world_pos = self._quat_rotate(qx, qy, qz, qw, cam_flu) + cached_t_world
                world_positions.append(world_pos)

            rr.log(
                "world/red_circles",
                rr.Points3D(world_positions, radii=[0.2],
                            colors=[[220, 50, 50]] * len(world_positions))
                if world_positions else rr.Points3D([]),
            )


with dai.Pipeline() as p:
    fps   = FPS
    width = WIDTH
    height = HEIGHT

    camRgb   = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    imu  = p.create(dai.node.IMU)
    odom = p.create(dai.node.BasaltVIO)
    slam = p.create(dai.node.RTABMapSLAM)

    slam.setParams({
        "RGBD/CreateOccupancyGrid": "true",
        "Grid/3D": "true",
        "Rtabmap/SaveWMState": "true",
        "RGBD/ProximityBySpace": "false",
    })

    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    slamStereo = p.create(dai.node.StereoDepth)
    slamStereo.setExtendedDisparity(False)
    slamStereo.setLeftRightCheck(True)
    slamStereo.setSubpixel(True)
    slamStereo.setRectifyEdgeFillColor(0)
    slamStereo.enableDistortionCorrection(True)
    slamStereo.initialConfig.setLeftRightCheckThreshold(10)
    slamStereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

    detStereo = p.create(dai.node.StereoDepth)
    detStereo.setExtendedDisparity(False)
    detStereo.setLeftRightCheck(True)
    detStereo.setSubpixel(False)
    detStereo.setRectifyEdgeFillColor(0)
    detStereo.enableDistortionCorrection(True)
    detStereo.initialConfig.setLeftRightCheckThreshold(10)
    detStereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    leftOut  = monoLeft.requestOutput((width, height))
    rightOut = monoRight.requestOutput((width, height))
    leftOut.link(slamStereo.left)
    rightOut.link(slamStereo.right)
    leftOut.link(detStereo.left)
    rightOut.link(detStereo.right)

    slamStereo.syncedLeft.link(odom.left)
    slamStereo.syncedRight.link(odom.right)
    slamStereo.depth.link(slam.depth)
    slamStereo.rectifiedLeft.link(slam.rect)
    imu.out.link(odom.imu)
    odom.transform.link(slam.odom)

    colorOut = camRgb.requestOutput((width, height))

    viewer = p.create(CircleSlamViewer)
    slam.transform.link(viewer.inputTrans)
    slam.passthroughRect.link(viewer.inputImg)
    slam.occupancyGridMap.link(viewer.inputGrid)
    slam.obstaclePCL.link(viewer.inputObstaclePCL)
    slam.groundPCL.link(viewer.inputGroundPCL)
    colorOut.link(viewer.inputColorImg)
    detStereo.depth.link(viewer.inputDepth)

    p.start()
    while p.isRunning():
        time.sleep(1)
