import depthai as dai


STEREO_DEFAULT_FPS = 20


class CameraBundle:
    """Helper to create and link camera nodes and stereo outputs."""

    def __init__(
        self,
        pipeline: dai.Pipeline,
        rgb_socket=dai.CameraBoardSocket.CAM_A,
        left_socket=dai.CameraBoardSocket.CAM_B,
        right_socket=dai.CameraBoardSocket.CAM_C,
        mono_resolution=(640, 320),
        sensor_fps: int = STEREO_DEFAULT_FPS,
    ):
        self.pipeline = pipeline
        self.camRgb = pipeline.create(dai.node.Camera).build(
            rgb_socket, sensorFps=sensor_fps
        )
        set_preview_keep_aspect_ratio = getattr(
            self.camRgb, "setPreviewKeepAspectRatio", None
        )
        if callable(set_preview_keep_aspect_ratio):
            set_preview_keep_aspect_ratio(False)
        self.monoLeft = pipeline.create(dai.node.Camera).build(
            left_socket, sensorFps=sensor_fps
        )
        self.monoRight = pipeline.create(dai.node.Camera).build(
            right_socket, sensorFps=sensor_fps
        )
        self.slam = pipeline.create(dai.node.RTABMapSLAM)

        self.stereo = pipeline.create(dai.node.StereoDepth)
        self.leftOutput = self.monoLeft.requestOutput(mono_resolution)
        self.rightOutput = self.monoRight.requestOutput(mono_resolution)
        self.leftOutput.link(self.stereo.left)
        self.rightOutput.link(self.stereo.right)
