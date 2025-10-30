import sys
import os
import math
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
import hybrid_control as hc
from cmd_src import CommandSource

class AnimatedStateControl(CommandSource):
    def __init__(self, robot, keyframes):
        """
        Args:
            robot: Robot instance
            keyframes: List of dicts with:
                - 'state': joint state dict (like carry_state, raise_state, etc.)
                - 'progress_joints': list of joint names to track progress (defaults to all joints)
                - 'zero_point': progress value where authority starts (0.0)
                - 'one_point': progress value where authority reaches 1.0 (1.0)
        """
        self.robot = robot
        self.forward_keyframes = keyframes
        self.reverse_keyframes = self.reverse_sequence(keyframes)
        self.reverse = False  # Default to forward sequence
        
        # Create StateControl for forward keyframes
        self.forward_controllers = []
        for frame in self.forward_keyframes:
            controller = StateControl(robot, frame['state'])
            self.forward_controllers.append(controller)
            
        # Create StateControl for reverse keyframes
        self.reverse_controllers = []
        for frame in self.reverse_keyframes:
            controller = StateControl(robot, frame['state'])
            self.reverse_controllers.append(controller)
    
    def get_interpolated_authority(self, current_value, zero_point, one_point):
        if zero_point > one_point:
            progress = (zero_point - current_value) / (zero_point - one_point)
        else:
            progress = (current_value - zero_point) / (one_point - zero_point)
        return max(0.0, min(1.0, progress))
    
    @staticmethod
    def reverse_sequence(sequence):
        """Generate reverse sequence from original sequence"""
        if len(sequence) < 2:
            return sequence
        
        reversed_frames = []
        
        # Reverse the order of frames
        for i, frame in enumerate(reversed(sequence)):
            new_frame = {'state': frame['state']}
            
            if i == 0:
                # First frame of reverse (was last frame): keep authority settings, no progress_joints
                if 'zero_point' in frame:
                    new_frame['zero_point'] = frame['zero_point']
                if 'one_point' in frame:
                    new_frame['one_point'] = frame['one_point']
            elif i == len(sequence) - 1:
                # Last frame of reverse (was first frame): add authority settings, keep progress_joints from next frame
                next_original_frame = sequence[len(sequence) - i]  # Next frame in original sequence
                if 'progress_joints' in next_original_frame:
                    new_frame['progress_joints'] = next_original_frame['progress_joints']
                # Add default authority settings since original first frame didn't have them
                new_frame['zero_point'] = 0.8
                new_frame['one_point'] = 0.9
            else:
                # Middle frames: get progress_joints from next frame in reverse sequence
                next_reverse_idx = len(sequence) - i - 1  # Index in original sequence
                if next_reverse_idx < len(sequence) - 1:  # Not the first frame of original
                    next_original_frame = sequence[next_reverse_idx + 1]
                    if 'progress_joints' in next_original_frame:
                        new_frame['progress_joints'] = next_original_frame['progress_joints']
                
                # Keep authority settings from current frame
                if 'zero_point' in frame:
                    new_frame['zero_point'] = frame['zero_point']
                if 'one_point' in frame:
                    new_frame['one_point'] = frame['one_point']
            
            reversed_frames.append(new_frame)
        
        return reversed_frames

    def get_command(self):
        """Return blended commands from all keyframes based on their authority"""
        # Choose which sequence to use based on reverse flag
        keyframes = self.reverse_keyframes if self.reverse else self.forward_keyframes
        controllers = self.reverse_controllers if self.reverse else self.forward_controllers
        
        cmd = {}
        
        for i, frame in enumerate(keyframes):
            controller = controllers[i]
            frame_cmd = controller.get_command()
            
            if i == 0:
                # First frame always has full authority initially
                cmd = hc.merge_override(frame_cmd, cmd)
            else:
                # Calculate progress from reference frame (i-2) to previous frame (i-1)
                # The previous frame's progress determines this frame's authority
                prev_frame = keyframes[i-1]
                prev_controller = controllers[i-1]
                
                # Reference state is the frame before the previous frame
                reference_frame = keyframes[i-2] if i >= 2 else keyframes[0]
                
                # Get progress joints from PREVIOUS frame (which joints to track)
                progress_joints = prev_frame.get('progress_joints', list(prev_frame['state'].keys()))
                
                # Previous controller calculates progress from reference state to its target
                progress_dict = prev_controller.get_progress(reference_frame['state'])
                
                # Average progress across specified joints
                joint_progresses = [progress_dict.get(joint, 0.0) for joint in progress_joints if joint in progress_dict]
                avg_progress = sum(joint_progresses) / len(joint_progresses) if joint_progresses else 0.0
                
                # Calculate authority based on progress (current frame needs zero_point/one_point)
                zero_point = frame.get('zero_point', 0.8)
                one_point = frame.get('one_point', 0.9)
                authority = self.get_interpolated_authority(avg_progress, zero_point, one_point)
                
                # Blend this frame's command
                cmd = hc.merge_mix(frame_cmd, cmd, authority)
        return cmd

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
            "gripper": 0.25          # rad -> normalized velocity
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
    
    def get_progress(self, previous_state):
        """Calculate progress from 0 to 1 from previous_state to desired_state"""
        current_state = self.get_current_state()
        progress = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state and joint in previous_state:
                current_pos = current_state[joint]
                prev_pos = previous_state[joint]
                
                total_distance = abs(desired_pos - prev_pos)
                distance_covered = abs(prev_pos - current_pos)
                progress[joint] = distance_covered / total_distance if total_distance > 0 else 1.0
        
        return progress
    
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

raise_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": math.radians(15),
    "wrist_yaw": math.radians(90),
    "lift": 1.15,
    "arm": 0.0,
    "gripper": math.radians(90),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

extend_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": math.radians(15),
    "wrist_yaw": math.radians(0),
    "lift": 1.15,
    "arm": 0.5,
    "gripper": math.radians(90),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

drop_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": math.radians(15),
    "wrist_yaw": math.radians(0),
    "lift": 1.15,
    "arm": 0.5,
    "gripper": math.radians(360),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

drop_sequence = [
    {
        'state': carry_state,
        # First frame: just reference state, no settings needed
    },
    {
        'state': raise_state,
        'progress_joints': ['lift'],  # Track lift progress for next frame's authority
        'zero_point': 0.8,           # Authority settings for this frame
        'one_point': 0.9
    },
    {
        'state': extend_state,
        'progress_joints': ['arm'],   # Track arm progress for next frame's authority
        'zero_point': 0.8,           # Authority settings for this frame
        'one_point': 0.9
    },
    {
        'state': drop_state,
        # Last frame: only authority settings (no next frame to trigger)
        'zero_point': 0.8,
        'one_point': 0.9
    },
]

# Generate reverse sequence automatically
rev_drop_sequence = AnimatedStateControl.reverse_sequence(drop_sequence)


if __name__ == "__main__":
    import stretch_body.robot as rb
    import hybrid_control as hc
    try:    
        # Initialize robot
        robot = rb.Robot()
        robot.startup()
        
        # Create animated state controller with drop sequence
        ac = AnimatedStateControl(robot, keyframes=drop_sequence)

        # Create velocity controller
        controller = NormalizedVelocityControl(robot)
        
        print("Running forward sequence for 10 seconds...")
        start_time = time.time()
        
        while True:
            # Switch to reverse after 10 seconds for demo
            if time.time() - start_time > 10 and not ac.reverse:
                print("Switching to reverse sequence...")
                ac.reverse = True
            
            # Get algorithmic command
            algo_cmd = ac.get_command()

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
