import depthai as dai
from pathlib import Path
from typing import Optional
from airside.detection.oakd.camera_bundle import CameraBundle


# Create pipeline
def add_basalt_vio_rtab(p: dai.Pipeline, cameras: Optional[CameraBundle] = None):
    with p:
        # Fetch output folder path and create it if it doesn't exist
        output_folder = Path(__file__).parent.parent.parent.parent.parent / "outputs"
        output_folder.mkdir(exist_ok=True)

        # Ensure we have a CameraBundle (created only if not provided)
        cameras = cameras or CameraBundle(p)
        imu = p.create(dai.node.IMU)
        odom = p.create(dai.node.BasaltVIO)
        stereo = cameras.stereo
        slam = cameras.slam
        slam.setDatabasePath(str(output_folder / "building.db"))
        params = {
            "RGBD/CreateOccupancyGrid": "true",
            "Grid/3D": "true",
            "Rtabmap/SaveWMState": "true",
            "Grid/CellSize": "0.05",
        }
        slam.setParams(params)

        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200
        )
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        stereo.setExtendedDisparity(True)

        stereo.syncedLeft.link(odom.left)
        stereo.syncedRight.link(odom.right)
        stereo.depth.link(slam.depth)
        stereo.rectifiedLeft.link(slam.rect)
        imu.out.link(odom.imu)

        odom.transform.link(slam.odom)
