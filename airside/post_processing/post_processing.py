from typing import Tuple

from util import Plane, Vector3d, Target, MappedTarget, Direction

def run(db_path: str, targets_path: str, mav_comm) -> None:
    def _generate_ply(db_path: str) -> str:
        return "pointcloud.ply"

    def _fit_planes(ply_path: str) -> list[Plane]:
        return []
    
    def _get_targets(targets_path: str) -> Tuple[list[Target], Direction]:
        return [], Direction.NORTH
    
    def _locate_targets(planes: list[Plane], targets: list[Target], first_direction: Direction) -> list[MappedTarget]:
        return []
    
    ply_path = _generate_ply(db_path)

    planes = _fit_planes(ply_path)

    targets, first_direction = _get_targets(targets_path)

    mapped_targets = _locate_targets(planes, targets, first_direction)

    mav_comm.send_mapped_targets(mapped_targets)
