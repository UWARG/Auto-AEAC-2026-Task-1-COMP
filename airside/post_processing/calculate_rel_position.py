
import util as util
import math
import numpy as np

class target_positions:
    DEVIATION=0.01
    ANGLE_THRESHOLD=0.087 #~=5 degrees
    def calculate(self,
                  walls:list[util.Plane],
                  ground:util.Plane,
                  targets:list[util.Target],
                  directions:util.Direction) -> list:
        bottom_corners=[]
        parallel_to_z={
             "perp1":None,
             "perp2":None,
             "parallel1":None,
             "parallel2":None
        }
        ground_vector=np.array([ground.normal.x,
                        ground.normal.z,
                        -ground.normal.y]
                        )
        parallel_to_z["parallel1"]=np.array([walls[0].normal.x,
                            walls[0].normal.z,
                            -walls[0].normal.y])
        for j in walls[1:]:
            wall2=np.array([j.normal.x,
                            j.normal.z,
                            -j.normal.y])
            dp=np.dot(parallel_to_z["parallel1"],wall2)
            angle=math.acos(dp/(np.linalg.norm(parallel_to_z["parallel1"])*np.linalg.norm(wall2)))
            if (abs(angle)>(math.pi/2+self.ANGLE_THRESHOLD)) or (abs(angle)<self.ANGLE_THRESHOLD):
                parallel_to_z["parallel2"]=j
                continue
            matrix=np.vstack((parallel_to_z["parallel1"],wall2,ground_vector))
            b=np.array([walls[0].offset,j.offset,ground.offset])
            corner=np.linalg.solve(matrix,b)
            print(corner)
            if not parallel_to_z["parallel1"]:
                 parallel_to_z["parallel1"]=j
            else:
                 parallel_to_z["parallel2"]=j
            bottom_corners.append(corner)
        second_wall=parallel_to_z["parallel2"]
        perp_wall_1=parallel_to_z["perp1"]
        perp_wall_2=parallel_to_z["perp2"]

        parallel_to_z["perp1"]=np.array([perp_wall_1.normal.x,
                                     perp_wall_1.normal.z,
                                     -perp_wall_1.normal.y])
        parallel_to_z["perp2"]=np.array([perp_wall_2.normal.x,
                                     perp_wall_2.normal.z,
                                     -perp_wall_2.normal.y])
        parallel_to_z["parallel2"]=np.array([parallel_to_z["parallel2"].normal.x,
                               parallel_to_z["parallel2"].normal.z,
                               -parallel_to_z["parallel2"].normal.y])
        b=np.array([parallel_to_z["parallel2"].offset,perp_wall_1.offset,ground.offset])
        bottom_corners.append(np.linalg.solve(
             np.vstack((parallel_to_z["parallel2"],
                       parallel_to_z["perp1"],
                       ground_vector)),
             b
        ))
        b=np.array([second_wall.offset,perp_wall_2.offset,ground.offset])
        bottom_corners.append(np.linalg.solve(
             np.vstack((parallel_to_z["parallel2"],
                       parallel_to_z["perp2"],
                       ground_vector)),
             b
        ))

    
        
        


