import open3d as o3d
import numpy as np
import random
import os

def find_walls_force(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Loading {filename}...")
    pcd = o3d.io.read_point_cloud(filename)
    
    # Downsample (5cm)
    # print("Downsampling (5cm)...")
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Remove noise 
    # print("Removing noise...")
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    # pcd = pcd.select_by_index(ind)
   
    remaining_cloud = pcd
    planes_found = []
    
    #Parameters for plane-fitting 
    threshold = 0.1
    min_pts = 500
    
    # Horizontal Limit: Only allow 2 horizontal planes (Floor + Ceiling)
    # Once we find 2, we ignore all other horizontal noise layers.
    horizontal_count = 0
    max_horizontal = 0

    print("Running Iterative RANSAC...")

    while True:
        # standard RANSAC
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=2000
        )
        
        if len(inliers) < min_pts:
            break

        # Check Orientation
        [a, b, c, d] = plane_model
        
        # Check if a plane is horizontal (Floor/Ceiling)
        is_horizontal = abs(c) > 0.5 

        keep_plane = True
        
        if is_horizontal:
            if horizontal_count < max_horizontal:
                horizontal_count += 1
                type_name = "FLOOR/CEILING"
            else:
                keep_plane = False
                type_name = "Useless (Discarded)"
        else:
            type_name = "WALL"

        if keep_plane:
            print(f" Found {type_name}: {len(inliers)} points")
            
            plane_cloud = remaining_cloud.select_by_index(inliers)
            r = random.random()
            g = random.random()
            b = random.random()
            plane_cloud.paint_uniform_color([r, g, b])
            
            planes_found.append(plane_cloud)
        else:
            print(f" Removing {type_name}: {len(inliers)} points")

        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        
        if len(remaining_cloud.points) < min_pts:
            break
        
        if len(planes_found) > 10:
            break

    # Visualize
    print(f"Found {len(planes_found)} valid planes.")
    
    # Paint leftovers Grey
    remaining_cloud.paint_uniform_color([0.2, 0.2, 0.2]) 
    planes_found.append(remaining_cloud)
    
    o3d.visualization.draw_geometries(planes_found, 
                                      window_name="Floor Result",
                                      width=1024, height=768)

if __name__ == "__main__":
    find_walls_force("cloud_pillar.ply")
    
