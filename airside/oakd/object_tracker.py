#!/usr/bin/env python3

import depthai as dai
from camera_bundle import CameraBundle

fullFrameTracking = False

# Create pipeline


def add_object_tracker(p: dai.Pipeline, cameras: "CameraBundle" = None):

    cameras = cameras or CameraBundle(p)
    camRgb = cameras.camRgb
    stereo = cameras.stereo

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        camRgb, stereo, "yolov6-nano"
    )
    objectTracker = p.create(dai.node.ObjectTracker)

    spatialDetectionNetwork.setConfidenceThreshold(0.6)
    spatialDetectionNetwork.input.setQueueSize(1)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    objectTracker.setDetectionLabelsToTrack([0])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(
        dai.TrackerIdAssignmentPolicy.SMALLEST_ID
    )

    if fullFrameTracking:
        camRgb.requestFullResolutionOutput().link(objectTracker.inputTrackerFrame)
        # do not block the pipeline if it's too slow on full frame
        objectTracker.inputTrackerFrame.setBlocking(False)
        objectTracker.inputTrackerFrame.setMaxSize(1)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    # create output queues to read tracking results
    qTracklets = objectTracker.out.createOutputQueue(maxSize=4, blocking=False)
    qFrame = objectTracker.passthroughTrackerFrame.createOutputQueue(maxSize=4, blocking=False)
    return qTracklets, qFrame

