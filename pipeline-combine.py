import Basalt_VIO_RTab
from rerun_node import RerunNode
import object_tracker
import depthai as dai
import time
from cameraBundle import CameraBundle
# This is the main pipeline-combine.py file that integrates Basalt VIO with RTAB-Map SLAM


with dai.Pipeline() as pipeline:
    cameraBundle = CameraBundle(pipeline)
    object_tracker.add_object_tracker(pipeline)
    Basalt_VIO_RTab.add_basalt_vio_rtab(pipeline, cameraBundle)
    rerunViewer = RerunNode()
    slam = cameraBundle.slam
    slam.transform.link(rerunViewer.inputTrans)
    slam.passthroughRect.link(rerunViewer.inputImg)
    slam.occupancyGridMap.link(rerunViewer.inputGrid)
    slam.obstaclePCL.link(rerunViewer.inputObstaclePCL)
    slam.groundPCL.link(rerunViewer.inputGroundPCL)
    pipeline.start()
    try:
        while pipeline.isRunning():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
