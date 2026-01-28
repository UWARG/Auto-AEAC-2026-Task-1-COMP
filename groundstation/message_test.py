import time
import random
import groundstation.main as groundscript
from pymavlink import mavutil

CONNECTION_STRING="tcp:localhost:1400"
AIRSIDE_COMPONENT_ID = 191
NUM_MESSAGES=5
def main():
    connection=mavutil.mavlink_connection(CONNECTION_STRING, 
                                          source_system=1,
                                          source_component=AIRSIDE_COMPONENT_ID)
    time.sleep(1)
    connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                 mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                 0,
                                 0,
                                 0)
    for _ in range (NUM_MESSAGES):
        message=(f"{random.choice(list(groundscript.DIRECTIONS.keys()))}," #random direction
                 f"{random.choice(list(groundscript.COLORS.keys()))}," #random color
                 f"{1+random.random()*9}," #random number 1 to 10
                 f"{1+random.random()*9}") #random number 1 to 10
        encoded_message=message.encode()
        connection.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO,
                                       encoded_message)
        time.sleep(1)
    return 

if __name__=="__main__":
    main()
    print("done")


