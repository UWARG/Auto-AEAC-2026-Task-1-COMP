import depthai as dai
import sys

from pathlib import Path

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

import cv2
import numpy as np


class RerunNode(dai.node.ThreadedHostNode):
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.inputTrans = dai.Node.Input(self)
        self.inputImg = dai.Node.Input(self)
        self.inputObstaclePCL = dai.Node.Input(self)
        self.inputGroundPCL = dai.Node.Input(self)
        self.inputGrid = dai.Node.Input(self)
        self.inputTracklets = dai.Node.Input(self)
        self.positions = []
        self.fx = 400.0
        self.fy = 400.0
        self.intrinsicsSet = False

    @staticmethod
    def _quat_rotate(qx, qy, qz, qw, v):
        q = np.array([qx, qy, qz])
        t = 2.0 * np.cross(q, v)
        return v + qw * t + np.cross(q, t)

    def getFocalLengthFromImage(self, imgFrame):
        p = self.getParentPipeline()
        calibHandler = p.getDefaultDevice().readCalibration()
        intrinsics = calibHandler.getCameraIntrinsics(
            dai.CameraBoardSocket(imgFrame.getInstanceNum()),
            imgFrame.getWidth(),
            imgFrame.getHeight(),
        )
        self.fx = intrinsics[0][0]
        self.fy = intrinsics[1][1]
        self.intrinsicsSet = True

    def run(self):
        rr.init("", spawn=True)
        rr.log("world", rr.ViewCoordinates.FLU)
        rr.log("world/ground", rr.Boxes3D(half_sizes=[3.0, 3.0, 0.00001]))
        while self.isRunning():
            transData = self.inputTrans.get()
            imgFrame = self.inputImg.get()
            if not self.intrinsicsSet:
                self.getFocalLengthFromImage(imgFrame)
            pclObstData = self.inputObstaclePCL.tryGet()
            pclGrndData = self.inputGroundPCL.tryGet()
            mapData = self.inputGrid.tryGet()
            trackletsMsg = self.inputTracklets.tryGet()
            if transData is not None:
                trans = transData.getTranslation()
                quat = transData.getQuaternion()
                position = rr.datatypes.Vec3D([trans.x, trans.y, trans.z])
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=position,
                        rotation=rr.datatypes.Quaternion(
                            xyzw=[quat.qx, quat.qy, quat.qz, quat.qw]
                        ),
                    ),
                )
                self.positions.append(position)
                lineStrip = rr.components.LineStrip3D(self.positions)
                rr.log("world/trajectory", rr.LineStrips3D(lineStrip))
                rr.log(
                    "world/camera/image",
                    rr.Pinhole(
                        resolution=[imgFrame.getWidth(), imgFrame.getHeight()],
                        focal_length=[self.fx, self.fy],
                        camera_xyz=rr.ViewCoordinates.FLU,
                    ),
                )
                img = imgFrame.getCvFrame()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rr.log("world/camera/image/rgb", rr.Image(img))
                if pclObstData is not None:
                    points, colors = pclObstData.getPointsRGB()
                    rr.log(
                        "world/obstacle_pcl",
                        rr.Points3D(points, colors=colors, radii=[0.01]),
                    )
                if pclGrndData is not None:
                    points, colors = pclGrndData.getPointsRGB()
                    rr.log(
                        "world/ground_pcl",
                        rr.Points3D(points, colors=colors, radii=[0.01]),
                    )
                if mapData is not None:
                    rr.log("map", rr.Image(mapData.map.getCvFrame()))
                if trackletsMsg is not None and isinstance(trackletsMsg, dai.Tracklets):
                    world_positions = []
                    labels = []
                    t_world = np.array([trans.x, trans.y, trans.z])
                    for t in trackletsMsg.tracklets:
                        if t.status not in (dai.Tracklet.TrackingStatus.NEW, dai.Tracklet.TrackingStatus.TRACKED):
                            continue
                        if t.spatialCoordinates.z == 0:
                            continue
                        # OAK-D spatial coords are RDF (right, down, forward).
                        # SLAM quaternion operates on FLU (forward, left, up).
                        # Convert: FLU_X=RDF_Z, FLU_Y=-RDF_X, FLU_Z=-RDF_Y
                        cam_pos = np.array([
                            t.spatialCoordinates.z / 1000.0,
                            -t.spatialCoordinates.x / 1000.0,
                            -t.spatialCoordinates.y / 1000.0,
                        ])
                        world_pos = self._quat_rotate(quat.qx, quat.qy, quat.qz, quat.qw, cam_pos) + t_world
                        world_positions.append(world_pos)
                        labels.append(f"ID:{t.id}")
                    if world_positions:
                        rr.log(
                            "world/targets",
                            rr.Points3D(world_positions, radii=[0.3], colors=[[255, 50, 50]] * len(world_positions), labels=labels),
                        )
