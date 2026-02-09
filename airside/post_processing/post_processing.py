from typing import Tuple

from airside.mavlink_comm import MavlinkComm
import numpy as np
import math
from util import Plane, Vector3d, Target, MappedTarget, Direction, Coordinate


def run(db_path: str, targets_path: str, mav_comm: MavlinkComm) -> None:
    def _generate_ply(db_path: str) -> str:
        return "pointcloud.ply"

    def _fit_planes(ply_path: str) -> list[Plane]:
        return []

    def _get_targets(targets_path: str) -> Tuple[list[Target], Direction]:
        return [], Direction.NORTH

    def _locate_targets(
        planes: list[Plane], targets: list[Target], first_direction: Direction
    ) -> list[MappedTarget]:
        MARGIN=0.5
        ANGLE_THRESHOLD=0.087 #~= 5 degrees
        PARALLEL_THRESHOLD=0.5
        bottom_corners=[]
        parallel_to_x={
            "parallel1":None, #wall with cardinal direction, as first index in ordered dict
            "perp1":None,
            "perp2":None,
            "parallel2":None,
            "ground":None
        }
        direction=["NORTH","EAST","SOUTH","WEST","NORTH","EAST","SOUTH","WEST"]
        for i in planes:
            wall_normal=np.array([i.normal.x,i.normal.y,i.normal.z])
            if wall_normal[2]>MARGIN: #if normal has some z component 
                parallel_to_x["ground"]=i
                continue
            dp=np.dot(wall_normal,np.array([1,0,0]))
            angle=math.acos(np.clip(dp/(np.linalg.norm(wall_normal)*1),-1,1))
            if angle<ANGLE_THRESHOLD or angle>math.pi-ANGLE_THRESHOLD:
                if not parallel_to_x["parallel1"]:
                    parallel_to_x["parallel1"]=i
                else:
                    if abs(parallel_to_x["parallel1"].offset)>abs(i.offset):
                        parallel_to_x["parallel2"]=parallel_to_x["parallel1"]
                        parallel_to_x["parallel1"]=i
                    else:
                        parallel_to_x["parallel2"]=i
            #Want perp1 to be the wall in the positive y direction
            else:
                if not parallel_to_x["perp1"]:
                    parallel_to_x["perp1"]=i
                else:
                    if (parallel_to_x["perp1"].offset>i.offset):
                        parallel_to_x["perp2"]=i
                    else:
                        parallel_to_x["perp2"]=parallel_to_x["perp1"]
                        parallel_to_x["perp1"]=i

        
        ground_vector=np.array([parallel_to_x["ground"].normal.x,
                        parallel_to_x["ground"].normal.y,
                        parallel_to_x["ground"].normal.z]
                        )
        wall1=np.array([parallel_to_x["parallel1"].normal.x,
                            parallel_to_x["parallel1"].normal.y,
                            parallel_to_x["parallel1"].normal.z])
        far_wall=np.array([parallel_to_x["parallel2"].normal.x, #far wall parallel to first wall with cardinal direction
                            parallel_to_x["parallel2"].normal.y,
                            parallel_to_x["parallel2"].normal.z])
        for j in [parallel_to_x["perp1"],parallel_to_x["perp2"]]:
            wall2=np.array([j.normal.x,
                            j.normal.y,
                            j.normal.z])
            matrix=np.vstack((wall1,wall2,ground_vector))
            matrix2=np.vstack((far_wall,wall2,ground_vector))
            b=np.array([parallel_to_x["parallel1"].offset,j.offset,parallel_to_x["ground"].offset])
            b2=np.array([parallel_to_x["parallel2"].offset,j.offset,parallel_to_x["ground"].offset])
            corner=np.linalg.solve(matrix,b)
            corner2=np.linalg.solve(matrix2,b2)
            bottom_corners.append(corner) #front corner
            bottom_corners.append(corner2) #back corner
        #reorder corners, left most corner for plane with specified cardinal direction has larger y value
        
        if bottom_corners[0][1]<bottom_corners[2][1]:
            temp=bottom_corners[0]
            bottom_corners[0]=bottom_corners[2]
            bottom_corners[2]=temp
        #same thing for back corners
        if bottom_corners[1][1]<bottom_corners[3][1]:
            temp=bottom_corners[1]
            bottom_corners[1]=bottom_corners[3]
            bottom_corners[3]=temp
        
        target_locations=[]
        for i in targets:
            target_on_wall=False
            target=np.array([i.location.x,
                             i.location.y,
                             i.location.z])
            for j,k in zip(bottom_corners,list(parallel_to_x.values())[0:3]):
                vector=np.subtract(target,j)
                normal_vector=np.array([k.normal.x,
                                        k.normal.y,
                                        k.normal.z])
                dp=np.dot(vector,normal_vector)
                angle=math.acos(dp/(np.linalg.norm(vector)*np.linalg.norm(normal_vector)))
                if angle>math.pi/2-ANGLE_THRESHOLD and angle<math.pi/2+ANGLE_THRESHOLD:
                    if vector[0]<PARALLEL_THRESHOLD:
                        right,up=(vector[1],vector[2])
                    if vector[1]<PARALLEL_THRESHOLD:
                        right,up=(vector[0],vector[2])
                    target_on_wall=True
                    index=direction.index(first_direction)
                    index2=list(direction).index(k)
                    wall_direction=direction[index+index2]
                    relative_position=Coordinate(right,0,up)
                    target_locations.append(MappedTarget(i.colour,relative_position,Direction[wall_direction],False))
                    break
            
                #value from right of bottom left corner will always be in +x or +y dir. Other component direction
                #should be 0 for "vector"
            #want to find corner regions 
            if not target_on_wall:
                wall=None
                if i.location.y>parallel_to_x["perp1"].offset/parallel_to_x["perp1"].normal.y:
                    if (i.location.x>parallel_to_x["parallel2"].offset/parallel_to_x["parallel2"].normal.x):
                        vector=np.subtract(target,bottom_corners[1]) #index for left corner of perp 1/ bottom right corner
                        position=(-abs(vector[0]),abs(vector[1]))
                        wall=parallel_to_x["perp1"]
                    elif (i.location.x<parallel_to_x["parallel1"].offset/parallel_to_x["parallel1"].normal.x):
                        vector=np.subtract(target,bottom_corners[0]) #left corner for parallel 1
                        position=(-abs(vector[1]),abs(vector[0]))
                        wall=parallel_to_x["parallel1"]
                    else:
                        vector=np.subtract(target,bottom_corners[1]) #left corner for perp 1
                        position=(abs(vector[0]),abs(vector[1])) 
                        wall=parallel_to_x["perp1"]

                elif i.location.y<parallel_to_x["perp2"].offset/parallel_to_x["perp2"].normal.y:
                    if (i.location.x>parallel_to_x["parallel2"].offset/parallel_to_x["parallel2"].normal.x):
                        vector=np.subtract(target,bottom_corners[2]) #left corner for perp 2
                        position=(-abs(vector[0]),abs(vector[1]))
                        wall=parallel_to_x["perp2"]
                    elif (i.location.x<parallel_to_x["parallel1"].offset/parallel_to_x["parallel1"].normal.x):
                        vector=np.subtract(target,bottom_corners[3]) #left corner for parallel 2
                        position=(-abs(vector[1]),abs(vector[0]))
                        wall=parallel_to_x["parallel2"]
                    else:
                        vector=np.subtract(target,bottom_corners[2]) 
                        position=(abs(vector[0]),abs(vector[1]))
                        wall=parallel_to_x["perp2"]
                #for in between two perpendicular walls
                else:
                    if (i.location.x>parallel_to_x["parallel2"]/parallel_to_x["parallel2"].normal.x):
                        vector=np.subtract(target,bottom_corners[3])
                        wall=parallel_to_x["parallel2"]
                    elif (i.location.x<parallel_to_x["parallel1"]/parallel_to_x["parallel1"].normal.x):
                        vector=np.subtract(target,bottom_corners[0])
                        wall=parallel_to_x["parallel1"]
                    position=(abs(vector[1]),abs(vector[0]))
                
                #create new dictionary
                index=direction.index(first_direction)
                index2=list(parallel_to_x).index(wall)
                wall_direction=direction[index+index2]
                relative_position=Coordinate(*position,0)
                target_locations.append(MappedTarget(i.colour,relative_position,Direction[wall_direction],False))



        return target_locations

    ply_path = _generate_ply(db_path)

    planes = _fit_planes(ply_path)

    targets, first_direction = _get_targets(targets_path)

    mapped_targets = _locate_targets(planes, targets, first_direction)

    for mapped_target in mapped_targets:
        mav_comm.send_mapped_target(mapped_target)
