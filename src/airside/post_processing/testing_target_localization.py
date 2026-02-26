from airside.post_processing import target_rel_position
from util import Plane, Vector3d, Target, MappedTarget, Direction, Coordinate, Colours

# Testing values 1
planes = []
planes.append(Plane(Vector3d(x=1, y=0, z=0), 2))  # x_near
planes.append(Plane(Vector3d(x=1, y=0, z=0), 7.3))  # x_far
planes.append(Plane(Vector3d(x=0, y=1, z=0), 2))  # y_positive
planes.append(Plane(Vector3d(x=0, y=-1, z=0), 2))  # y_negative
planes.append(Plane(Vector3d(x=0, y=0, z=1), 3))  # ground

targets = []
targets.append(Target(colour=Colours.BLACK, location=Coordinate(x=4, y=2.1, z=0)))
targets.append(Target(colour=Colours.BLUE, location=Coordinate(x=6.7, y=2.1, z=-2)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=2, y=2, z=0)))
targets.append(Target(colour=Colours.YELLOW, location=Coordinate(x=7.3, y=-2, z=0)))
targets.append(Target(colour=Colours.YELLOW, location=Coordinate(x=7.3, y=0, z=0)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=5, y=-2, z=0)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=7.3, y=0, z=0)))
# floor targets
targets.append(Target(colour=Colours.RED, location=Coordinate(x=8, y=2.5, z=3)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=7.2, y=2.5, z=3)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=7.1, y=1.9, z=3)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=8.1, y=2.5, z=3)))
targets.append(Target(colour=Colours.RED, location=Coordinate(x=0, y=-2.5, z=3)))


MappedTarget = target_rel_position.locate_targets(
    planes=planes, targets=targets, first_direction=Direction.WEST
)
for i in MappedTarget:
    print(i)
