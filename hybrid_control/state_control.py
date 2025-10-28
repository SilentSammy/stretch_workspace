import sys
import os
import math
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
from cmd_src import CommandSource

class StateControl(CommandSource):
    def __init__(self, robot, desired_state):
        self.robot = robot
        self.desired_state = desired_state
        
        # Individual Kp values for different joint types
        self.Kp = {
            "wrist_roll": 1.0,       # rad -> normalized velocity
            "wrist_pitch": 1.0,      # rad -> normalized velocity
            "wrist_yaw": 1.0,        # rad -> normalized velocity
            "lift": 3.0,             # m -> normalized velocity
            "arm": 5.0,              # m -> normalized velocity
            "head_pan": 1.0,         # rad -> normalized velocity
            "head_tilt": 1.0,        # rad -> normalized velocity
            "gripper": 0.25           # rad -> normalized velocity
        }
        
        # Maximum velocity limits (overrides default 1.0)
        self.max_velocity = {
            "lift": 0.75,            # Limit lift to 75% max speed
            # Add other joint-specific limits here as needed
        }
        
        self.tolerance = {
            "wrist_roll": 0.02,      # rad
            "wrist_pitch": 0.02,     # rad  
            "wrist_yaw": 0.02,       # rad
            "lift": 0.01,            # m
            "arm": 0.01,             # m
            "head_pan": 0.02,        # rad
            "head_tilt": 0.02,       # rad
            "gripper": 0.1           # rad
        }
    
    def get_current_state(self):
        """Get current joint positions from robot"""
        current_state = {}
        
        if "arm" in self.desired_state:
            current_state["arm"] = self.robot.arm.status['pos']
        if "lift" in self.desired_state:
            current_state["lift"] = self.robot.lift.status['pos']
        if "wrist_roll" in self.desired_state:
            current_state["wrist_roll"] = self.robot.end_of_arm.motors['wrist_roll'].status['pos']
        if "wrist_pitch" in self.desired_state:
            current_state["wrist_pitch"] = self.robot.end_of_arm.motors['wrist_pitch'].status['pos']
        if "wrist_yaw" in self.desired_state:
            current_state["wrist_yaw"] = self.robot.end_of_arm.motors['wrist_yaw'].status['pos']
        if "head_pan" in self.desired_state:
            current_state["head_pan"] = self.robot.head.status['head_pan']['pos']
        if "head_tilt" in self.desired_state:
            current_state["head_tilt"] = self.robot.head.status['head_tilt']['pos']
        if "gripper" in self.desired_state:
            current_state["gripper"] = self.robot.end_of_arm.motors['stretch_gripper'].status['pos']
            
        return current_state
    
    def is_at_goal(self):
        """Check if robot is within tolerance of desired state"""
        current_state = self.get_current_state()
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                error = abs(current_state[joint] - desired_pos)
                if error > self.tolerance.get(joint, 0.01):
                    return False
        return True
    
    def get_command(self):
        """Return proportional velocity commands to reach desired state"""
        current_state = self.get_current_state()
        command = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                # Calculate position error
                error = desired_pos - current_state[joint]
                
                # Set to zero velocity if within tolerance
                if abs(error) <= self.tolerance.get(joint, 0.01):
                    velocity = 0.0
                else:
                    # Calculate proportional velocity using joint-specific Kp
                    kp = self.Kp.get(joint, 1.0)
                    velocity = kp * error
                    
                    # Clamp velocity to joint-specific max (default 1.0)
                    max_vel = self.max_velocity.get(joint, 1.0)
                    velocity = max(-max_vel, min(max_vel, velocity))
                
                # Map to normalized velocity command format
                if joint == "arm":
                    command["arm_out"] = velocity
                elif joint == "lift":
                    command["lift_up"] = velocity
                elif joint == "wrist_roll":
                    command["wrist_roll_counterclockwise"] = velocity
                elif joint == "wrist_pitch":
                    command["wrist_pitch_up"] = velocity
                elif joint == "wrist_yaw":
                    command["wrist_yaw_counterclockwise"] = velocity
                elif joint == "head_pan":
                    command["head_pan_counterclockwise"] = velocity
                elif joint == "head_tilt":
                    command["head_tilt_up"] = velocity
                elif joint == "gripper":
                    command["gripper_open"] = velocity
        
        return command

stowed_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": -math.radians(15),
    "wrist_yaw": math.radians(90),
    "lift": 0.2,
    "arm": 0.0,
    "gripper": math.radians(360),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

carry_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": -math.radians(15),
    "wrist_yaw": math.radians(90),
    "lift": 0.2,
    "arm": 0.0,
    "gripper": math.radians(90),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

if __name__ == "__main__":
    import stretch_body.robot as rb
    import hybrid_control as hc
    try:    
        # Initialize robot
        robot = rb.Robot()
        robot.startup()
        
        # Create state controller
        sc = StateControl(robot, desired_state=carry_state)

        # Create velocity controller
        controller = NormalizedVelocityControl(robot)

        print("Moving to stowed position...")
        
        while True:
            # Get algorithmic command
            algo_cmd = sc.get_command()

            # Hybridize commands (could add human override here)
            hybrid_cmd = hc.hybridize(algo_cmd)

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
