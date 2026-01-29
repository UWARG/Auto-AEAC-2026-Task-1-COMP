import Basalt_VIO_RTab
import rerun_node
import object_tracker
import depthai as dai

# This is the main pipeline-combine.py file that integrates Basalt VIO with RTAB-Map SLAM


with dai.Pipeline() as pipeline:

    object_tracker.add_object_tracker(pipeline)
    rerunViewer = Basalt_VIO_RTab.add_basalt_vio_rtab(pipeline)
    pipeline.start()
    while not KeyboardInterrupt and pipeline.isRunning():
        pass
    pipeline.stop()
