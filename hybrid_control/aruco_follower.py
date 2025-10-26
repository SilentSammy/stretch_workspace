import sys
import os
import math
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
import stretch_body.robot as rb

def follow_aruco():
    cmd = {
        'head_pan_counterclockwise' : 0.5,
        'head_tilt_up':0.5,
    }
    return cmd

if __name__ == "__main__":
    import hybrid_control as hc
    try:    
        # Initialize robot
        robot = rb.Robot()
        robot.startup()
        
        # Create velocity controller
        controller = NormalizedVelocityControl(robot)

        print("Moving to stowed position...")
        
        while True:
            # Get algorithmic command
            aruco_cmd = follow_aruco()

            # Hybridize commands (could add human override here)
            hybrid_cmd = hc.hybridize(aruco_cmd)

            # Send command
            controller.set_command(hybrid_cmd)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.stop()
        if 'robot' in locals():
            robot.stop()
