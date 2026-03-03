import open3d as o3d
import numpy as np
import random
from airside.mavlink_comm import MavlinkComm

def is_coplanar(plane_model1, plane_model2, angle_tol_deg=15.0, dist_tol=0.3):
    """
    Checks if two plane equations represent the same physical plane
    """

    # Plane contains equation coefficients [a, b, c, d]
    # Extract a, b, and c for the normal vector
    n1 = np.array(plane_model1[:3])
    n2 = np.array(plane_model2[:3])

    # Extract d values for distance comparison
    d1 = plane_model1[3]
    d2 = plane_model2[3]

    # Set to unit vector
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    dot_prod = np.dot(n1, n2)

    # Check if the normals are facing opposite directions
    if dot_prod < 0:
        n2 = -n2
        d2 = -d2
        dot_prod = -dot_prod

    # Check if it is the two planes are within the angle tolerance to be considered parallel
    angle_tol = np.cos(np.deg2rad(angle_tol_deg))
    if dot_prod < angle_tol:
        return False

    # Check if the planes are within the distance tolerance to be considered the same plane
    if abs(d1 - d2) > dist_tol:
        return False

    return True


def find_walls(filename: str, mav_comm: MavlinkComm):

    mav_comm.info(f"Loading {filename}...")
    pcd = o3d.io.read_point_cloud(filename)

    # Downsample for speed
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    mav_comm.info("Filtering noise...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    remaining_cloud = pcd
    extracted_planes = []

    threshold = 0.1
    min_pts = 5000
    max_horizontal = 1
    horizontal_count = 0

    mav_comm.info("Running Iterative RANSAC...")

    while True:
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=5000
        )

        if len(inliers) < min_pts:
            break

        [a, b, c, d] = plane_model
        is_horizontal = abs(c) > 0.45

        if is_horizontal:
            if horizontal_count < max_horizontal:
                horizontal_count += 1
            else:
                pass
        else:
            plane_cloud = remaining_cloud.select_by_index(inliers)
            extracted_planes.append({"model": plane_model, "cloud": plane_cloud})

        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

        if len(remaining_cloud.points) < min_pts:
            break

    mav_comm.info(f"Found {len(extracted_planes)} planes.")
    mav_comm.info("Merging coplanar patches...")

    merged_walls = []

    for patch in extracted_planes:
        matched = False
        current_angle_tol = 25.0
        current_dist_tol = 3.0

        for wall_group in merged_walls:
            for existing_model in wall_group["models"]:
                if is_coplanar(
                    patch["model"],
                    existing_model,
                    angle_tol_deg=current_angle_tol,
                    dist_tol=current_dist_tol,
                ):
                    wall_group["clouds"].append(patch["cloud"])
                    wall_group["models"].append(patch["model"])
                    matched = True
                    break

            if matched:
                break

        if not matched:
            merged_walls.append(
                {"models": [patch["model"]], "clouds": [patch["cloud"]]}
            )

    mav_comm.info(f"Merged into {len(merged_walls)} distinct physical walls.")

    # Sort walls by size so the largest walls
    merged_walls.sort(
        key=lambda group: sum([len(cloud.points) for cloud in group["clouds"]]),
        reverse=True,
    )

    mav_comm.info("Detecting corners ...")

    adjusted_indices = set()
    corner_proximity_threshold = 3.0  # Maximum distance in units/meters between two walls to be considered a corner

    # Evaluate every possible pair of walls
    for i in range(len(merged_walls)):
        cloud_i = o3d.geometry.PointCloud()
        for c in merged_walls[i]["clouds"]:
            cloud_i += c

        n_i = np.array(merged_walls[i]["models"][0][:3])
        n_i = n_i / np.linalg.norm(n_i)

        for j in range(i + 1, len(merged_walls)):
            cloud_j = o3d.geometry.PointCloud()
            for c in merged_walls[j]["clouds"]:
                cloud_j += c

            n_j = np.array(merged_walls[j]["models"][0][:3])
            n_j = n_j / np.linalg.norm(n_j)

            dot_prod = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_prod))
            acute_angle = 180.0 - angle if angle > 90 else angle

            # If they are roughly perpendicular (between 60 and 120 degrees)
            if 60 < acute_angle < 120:
                # Check Physical Proximity
                dists = cloud_i.compute_point_cloud_distance(cloud_j)
                min_dist = np.min(dists) if len(dists) > 0 else float("inf")

                if min_dist < corner_proximity_threshold:

                    if j not in adjusted_indices:
                        n_j_ortho = n_j - (np.dot(n_j, n_i) * n_i)
                        n_j_ortho = n_j_ortho / np.linalg.norm(n_j_ortho)

                        center_j = cloud_j.get_center()
                        d_ortho = -np.dot(n_j_ortho, center_j)

                        merged_walls[j]["models"][0] = [
                            n_j_ortho[0],
                            n_j_ortho[1],
                            n_j_ortho[2],
                            d_ortho,
                        ]

                        # Flatten point cloud onto new plane
                        for cloud in merged_walls[j]["clouds"]:
                            points = np.asarray(cloud.points)
                            distances = np.dot(points, n_j_ortho) + d_ortho
                            projected_points = points - np.outer(distances, n_j_ortho)

                            micro_noise = np.random.normal(
                                0, 1e-5, projected_points.shape
                            )
                            cloud.points = o3d.utility.Vector3dVector(
                                projected_points + micro_noise
                            )

                        # Update normal for any future corner checks involving Wall J
                        n_j = n_j_ortho
                        adjusted_indices.add(j)

    walls = []
    for i, wall_group in enumerate(merged_walls):
        combined_cloud = o3d.geometry.PointCloud()
        for cloud in wall_group["clouds"]:
            combined_cloud += cloud

        # Diagnostic Output
        model = wall_group["models"][0]
        normal = np.array(model[:3])
        normal = normal / np.linalg.norm(normal)
        total_points = len(combined_cloud.points)

        mav_comm.info(
            f"Wall {i+1} | Points: {total_points} |Normal (X, Y, Z): [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]"
        )
        walls.append([normal[0], normal[1], normal[2]])

    return walls
