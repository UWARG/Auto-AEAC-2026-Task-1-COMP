import Basalt_VIO_RTab
from rerun_node import RerunNode
import object_tracker
import depthai as dai
import time
from camera_bundle import CameraBundle

# This is the main pipeline-combine.py file that integrates Basalt VIO with RTAB-Map SLAM


with dai.Pipeline() as pipeline:
    cameraBundle = CameraBundle(pipeline)
    qTracklets, qFrame = object_tracker.add_object_tracker(pipeline, cameraBundle)
    Basalt_VIO_RTab.add_basalt_vio_rtab(pipeline, cameraBundle)

    rerunViewer = RerunNode()
    slam = cameraBundle.slam
    slam.transform.link(rerunViewer.inputTrans)
    slam.passthroughRect.link(rerunViewer.inputImg)
    slam.occupancyGridMap.link(rerunViewer.inputGrid)
    slam.obstaclePCL.link(rerunViewer.inputObstaclePCL)
    slam.groundPCL.link(rerunViewer.inputGroundPCL)

    pipeline.start()
    last_frame_print = 0.0
    try:
        while pipeline.isRunning():
            # Get tracker outputs
            trackMsg = qTracklets.tryGet()
            frameMsg = qFrame.tryGet()
            # to confirm if frames are flowing or not
            if frameMsg and (time.time() - last_frame_print) > 1.0:
                print("Tracker frame received")
                last_frame_print = time.time()
            # print tracked targets only when being tracked    
            if trackMsg:
                for t in trackMsg.tracklets:
                    # print only when actively tracked to minimize  spam
                    if t.status.name in ("TRACKED", "NEW"):
                        sc = getattr(t, "spatialCoordinates", None)
                        if sc:
                            print(f"ID={t.id} status={t.status.name} X={sc.x} Y={sc.y} Z={sc.z} mm")
                        else:
                            print(f"ID={t.id} status={t.status.name}")
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()