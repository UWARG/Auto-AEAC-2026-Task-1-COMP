from airside.mavlink_comm import MavlinkComm
from util import Plane, Target, MappedTarget, Direction, Coordinate, Colours
from airside.post_processing import (
    target_rel_position,
    wall_detection,
    cluster_estimation,
)


def run(
    obstacle_pcl_path: str,
    ground_pcl_path: str,
    targets_path: str,
    mav_comm: MavlinkComm,
    first_direction: Direction,
) -> None:
    def _parse_target_line(line: str) -> Target | None:
        # Expected format from Target.__str__: "COLOUR, (x, y, z)"
        line = line.strip()
        if not line or "," not in line:
            return None

        colour_part, location_part = line.split(",", 1)
        colour_name = colour_part.strip()
        location_part = location_part.strip()

        if not (location_part.startswith("(") and location_part.endswith(")")):
            return None

        coords_raw = location_part[1:-1]
        coord_parts = [part.strip() for part in coords_raw.split(",")]
        if len(coord_parts) != 3:
            return None

        x_raw, y_raw, z_raw = coord_parts

        colour = Colours.__members__.get(colour_name.upper())
        if colour is None:
            return None

        try:
            location = Coordinate(float(x_raw), float(y_raw), float(z_raw))
        except ValueError:
            return None

        return Target(colour=colour, location=location)

    def _get_targets(targets_path: str) -> list[Target]:
        raw_targets: list[Target] = []
        skipped_lines = 0

        with open(targets_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed_target = _parse_target_line(line.strip())
                if parsed_target is None:
                    skipped_lines += 1
                    continue
                raw_targets.append(parsed_target)

        if skipped_lines > 0:
            mav_comm.logger.warning(
                f"Skipped {skipped_lines} unparsable target lines in {targets_path}"
            )

        if not raw_targets:
            mav_comm.logger.error("No targets found in target file")
            return []

        clustered_targets = cluster_estimation.cluster_estimation(raw_targets)

        if not clustered_targets:
            return raw_targets

        return clustered_targets

    def _fit_planes(obstacle_pcl_path: str, ground_pcl_path: str) -> list[Plane]:
        return wall_detection.find_walls(obstacle_pcl_path, ground_pcl_path)

    def _locate_targets(
        planes: list[Plane], targets: list[Target], first_direction: Direction
    ) -> list[MappedTarget]:
        return target_rel_position.locate_targets(planes, targets, first_direction)

    planes = _fit_planes(obstacle_pcl_path, ground_pcl_path)

    targets = _get_targets(targets_path)
    if not targets:
        mav_comm.logger.warning("No targets available for localization")
        return
    
    print("here0")

    mapped_targets = _locate_targets(planes, targets, first_direction)

    print("here1")

    for mapped_target in mapped_targets:
        mav_comm.send_mapped_target(mapped_target)

    print("here2")
