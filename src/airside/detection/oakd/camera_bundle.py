import depthai as dai


class CameraBundle:
    """Helper to create and link camera nodes and stereo outputs."""

    def __init__(
        self,
        pipeline: dai.Pipeline,
        rgb_socket=dai.CameraBoardSocket.CAM_A,
        left_socket=dai.CameraBoardSocket.CAM_B,
        right_socket=dai.CameraBoardSocket.CAM_C,
        mono_resolution=(640, 400),
    ):
        self.pipeline = pipeline
        self.camRgb = pipeline.create(dai.node.Camera).build(rgb_socket)
        self.monoLeft = pipeline.create(dai.node.Camera).build(left_socket)
        self.monoRight = pipeline.create(dai.node.Camera).build(right_socket)
        self.slam = pipeline.create(dai.node.RTABMapSLAM)

        self.stereo = pipeline.create(dai.node.StereoDepth)
        self.leftOutput = self.monoLeft.requestOutput(mono_resolution)
        self.rightOutput = self.monoRight.requestOutput(mono_resolution)
        self.leftOutput.link(self.stereo.left)
        self.rightOutput.link(self.stereo.right)
