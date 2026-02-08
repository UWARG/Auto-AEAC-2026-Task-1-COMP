
import util as util
import math
import numpy as np
"""
The leftmost corner of the first plane should have the largest positive y value.
All other corners can be determined based off this information of the first bottom left corner

"""
class target_positions:
    ANGLE_THRESHOLD=0.087 #~=5 degrees
    PARALLEL_THRESHOLD=0.5
    def calculate(self,
                  walls:list[util.Plane],
                  ground:util.Plane,
                  targets:list[util.Target],
                  directions:util.Direction) -> list:
        bottom_corners=[]
        parallel_to_x={
            "parallel1":None, #wall with cardinal direction, as first index in ordered dict
            "perp1":None,
            "perp2":None,
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
            #Want perp1 to be the wall in the positive y direction
            else:
                if not parallel_to_x["perp1"]:
                    parallel_to_x["perp1"]=i
                else:
                    if (parallel_to_x["perp1"].normal.y>i.normal.y):
                        parallel_to_x["perp2"]=i
                    else:
                        parallel_to_x["perp2"]=parallel_to_x["perp1"]
                        parallel_to_x["perp1"]=i

        
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
            bottom_corners.append(corner) #front corner
            bottom_corners.append(corner2) #back corner
        #reorder corners, left most corner for plane with specified cardinal direction has larger y value
        
        if bottom_corners[0].tolist()[1]<bottom_corners[2].tolist()[1]:
            temp=bottom_corners[0]
            bottom_corners[0]=bottom_corners[2]
            bottom_corners[2]=temp
        #same thing for back corners
        if bottom_corners[1].tolist()[1]<bottom_corners[3].tolist()[1]:
            temp=bottom_corners[1]
            bottom_corners[1]=bottom_corners[3]
            bottom_corners[3]=temp
        
        for i in targets:
            target=np.array([i.location.x,
                             i.location.y,
                             i.location.z])
            for j,k in zip(bottom_corners,parallel_to_x):
                vector=np.subtract(target,j)
                normal_vector=np.array([k.normal.x,
                                        k.normal.y,
                                        k.normal.z])
                dp=np.dot(vector,normal_vector)
                angle=math.acos(dp/(np.linalg.norm(vector)*np.linalg.norm(normal_vector)))
                if angle>math.pi/2-self.ANGLE_THRESHOLD and angle<math.pi/2+self.ANGLE_THRESHOLD:
                    if vector.tolist()[0]<self.PARALLEL_THRESHOLD:
                        (right,up)=(vector.tolist()[1],vector.tolist()[2])
                    if vector.tolist()[1]<self.PARALLEL_THRESHOLD:
                        (right,up)=(vector.tolist()[0],vector.tolist()[2])
                    target_on_wall=True
                    break
                #value from right of bottom left corner will always be in +x or +y dir. Other component direction
                #should be 0 for "vector"
            if not target_on_wall:
                vector=np.subtract(target,bottom_corners[0]) # can be any corner, doesn't matter
                dp=np.dot(ground_vector,vector)
                angle=math.acos(dp/(np.linalg.norm(vector)*np.linalg.norm(ground_vector)))
                if angle>math.pi/2-self.ANGLE_THRESHOLD and angle<math.pi/2+self.ANGLE_THRESHOLD:
                    
