
import util as util
import math
import numpy as np
"""
The leftmost corner of the first plane should have the largest positive y value.
All other corners can be determined based off this information of the first bottom left corner

"""
class target_positions:
    DEVIATION=0.01
    ANGLE_THRESHOLD=0.087 #~=5 degrees
    def calculate(self,
                  walls:list[util.Plane],
                  ground:util.Plane,
                  targets:list[util.Target],
                  directions:util.Direction) -> list:
        bottom_corners=[]
        parallel_to_x={
             "perp1":None,
             "perp2":None,
             "parallel1":None, # first wall with cardinal direction
             "parallel2":None
        }
        for i in walls:
            wall_normal=np.array([i.normal.x,i.normal.y,i.normal.z])
            dp=np.dot(wall_normal,np.array([1,0,0]))
            angle=math.acos(np.clip(dp/(np.linalg.norm(wall_normal)*1),-1,1))
            if angle<self.ANGLE_THRESHOLD or angle>math.pi-self.ANGLE_THRESHOLD:
                if not parallel_to_x["parallel1"]:
                    parallel_to_x["parallel1"]=i
                else:
                    if abs(parallel_to_x["parallel1"].offset)>abs(i.offset):
                        parallel_to_x["parallel2"]=parallel_to_x["parallel1"]
                        parallel_to_x["parallel1"]=i
                    else:
                        parallel_to_x["parallel2"]=i
            else:
                if not parallel_to_x["perp1"]:
                    parallel_to_x["perp1"]=i
                else:
                    parallel_to_x["perp2"]=i

        
        ground_vector=np.array([ground.normal.x,
                        ground.normal.y,
                        ground.normal.z]
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
            b=np.array([parallel_to_x["parallel1"].offset,j.offset,ground.offset])
            b2=np.array([parallel_to_x["parallel2"].offset,j.offset,ground.offset])
            corner=np.linalg.solve(matrix,b)
            corner2=np.linalg.solve(matrix2,b2)
            bottom_corners.append(corner) #front 2 corners
            bottom_corners.append(corner2) #back 2 corners


        
            
            

    
        
        


