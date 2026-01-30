import depthai as dai
from cameraBundle import CameraBundle


# Create pipeline
def add_basalt_vio_rtab(p: dai.Pipeline, cameras: "CameraBundle" = None):
    with p:
        # Ensure we have a CameraBundle (created only if not provided)
        cameras = cameras or CameraBundle(p)
        fps = 60
        width = 640
        height = 400
        # Define sources and outputs
        
        left = cameras.monoLeft
        right = cameras.monoRight
        imu = p.create(dai.node.IMU)
        odom = p.create(dai.node.BasaltVIO)
        slam = cameras.slam
        stereo = cameras.stereo
        params = {
            "RGBD/CreateOccupancyGrid": "true",
            "Grid/3D": "true",
            "Rtabmap/SaveWMState": "true",
        }
        slam.setParams(params)

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

        left.requestOutput((width, height)).link(stereo.left)
        right.requestOutput((width, height)).link(stereo.right)
        stereo.syncedLeft.link(odom.left)
        stereo.syncedRight.link(odom.right)
        stereo.depth.link(slam.depth)
        stereo.rectifiedLeft.link(slam.rect)
        imu.out.link(odom.imu)

        odom.transform.link(slam.odom)
        
