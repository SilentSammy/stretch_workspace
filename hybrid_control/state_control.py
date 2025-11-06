import sys
import os
import math
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
import hybrid_control as hc
import stretch_body.robot as rb
import hybrid_control as hc
from cmd_src import CommandSource

def get_interpolated_authority(current_value, zero_point, one_point):
    if zero_point > one_point:
        progress = (zero_point - current_value) / (zero_point - one_point)
    else:
        progress = (current_value - zero_point) / (one_point - zero_point)
    return max(0.0, min(1.0, progress))
    
class AnimatedStateControl(CommandSource):
    def __init__(self, robot, keyframes):
        self.frames = None
        self.rev_frames = None
        self.setup_frames(keyframes)
        self.robot = robot
        self.reverse = False  # Default to forward sequence

    def setup_frames(self, keyframes):
        self.frames = keyframes.copy()
        self.rev_frames = self.reverse_sequence(keyframes)

        for frames in [self.frames, self.rev_frames]:
            # Create controllers for each state in test sequence
            for i in range(1, len(frames)):
                frames[i]['controller'] = StateControl(robot, frames[i]['state'])
                frames[i]['progress'] = 0.0

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
                # First frame of reverse (was last frame): keep authority settings, no ref_joint
                if 'zero_point' in frame:
                    new_frame['zero_point'] = frame['zero_point']
                if 'one_point' in frame:
                    new_frame['one_point'] = frame['one_point']
            elif i == len(sequence) - 1:
                # Last frame of reverse (was first frame): add authority settings, get ref_joint from next frame
                next_original_frame = sequence[len(sequence) - i]  # Next frame in original sequence
                if 'ref_joint' in next_original_frame:
                    new_frame['ref_joint'] = next_original_frame['ref_joint']
                # Add default authority settings since original first frame didn't have them
                new_frame['zero_point'] = 0.8
                new_frame['one_point'] = 0.9
            else:
                # Middle frames: get ref_joint from next frame in reverse sequence
                next_reverse_idx = len(sequence) - i - 1  # Index in original sequence
                if next_reverse_idx < len(sequence) - 1:  # Not the first frame of original
                    next_original_frame = sequence[next_reverse_idx + 1]
                    if 'ref_joint' in next_original_frame:
                        new_frame['ref_joint'] = next_original_frame['ref_joint']
                
                # Keep authority settings from current frame
                if 'zero_point' in frame:
                    new_frame['zero_point'] = frame['zero_point']
                if 'one_point' in frame:
                    new_frame['one_point'] = frame['one_point']
            
            reversed_frames.append(new_frame)
        
        return reversed_frames

    def update_progress(self):
        # Calculate the progress for each frame in the test sequence
        frames = self.frames if not self.reverse else self.rev_frames
        for i in range(1, len(frames)):
            frame = frames[i]

            # Calculate the progress relative to previous state
            prev_state = frames[i-1]['state']
            ref_joint = frame['ref_joint']
            progress_dict = frame['controller'].get_progress(prev_state)
            progress = progress_dict[ref_joint]
            frame['progress'] = progress

    def get_command(self):
        self.update_progress()

        frames = self.frames if not self.reverse else self.rev_frames

        # Blend commands based on progress and authority
        acum_auth = 1.0
        cmd = frames[1]['controller'].get_command() # Start with 2nd frame command as base
        for i in range(2, len(frames)): # Iterate remaining frames
            frame = frames[i]

            # Calculate this frame's authority based on previous frame's progress
            prev_prog = frames[i-1]['progress']
            zero_point = frame['zero_point']
            one_point = frame['one_point']
            authority = get_interpolated_authority(prev_prog, zero_point, one_point)
            acum_auth *= authority

            # Merge this frame's command
            frame_cmd = frame['controller'].get_command()
            cmd = hc.merge_mix(frame_cmd, cmd, acum_auth)
        
        return cmd

    def get_progress(self):
        self.update_progress()
        frames = self.frames if not self.reverse else self.rev_frames
        progresses = [frame.get('progress', 0.0) for frame in frames[1:]]
        if not progresses:
            return 0.0
        return sum(progresses) / len(progresses)

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
    "lift": 1.1,
    "arm": 0.0,
    "gripper": math.radians(90),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

extend_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": math.radians(15),
    "wrist_yaw": math.radians(0),
    "lift": 1.1,
    "arm": 0.5,
    "gripper": math.radians(90),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

drop_state = {
    "wrist_roll": 0.0,
    "wrist_pitch": math.radians(15),
    "wrist_yaw": math.radians(0),
    "lift": 1.1,
    "arm": 0.5,
    "gripper": math.radians(360),
    "head_pan": 0.0,
    "head_tilt": 0.0,
}

drop_seq = [
    {
        'state': carry_state,
    },
    {
        'state': raise_state,
        'ref_joint': 'lift',
        'zero_point': 0.8,
        'one_point': 0.9
    },
    {
        'state': extend_state,
        'ref_joint': 'arm',
        'zero_point': 0.8,
        'one_point': 0.9
    },
    {
        'state': drop_state,
        'ref_joint': 'gripper',
        'zero_point': 0.8,
        'one_point': 0.9
    }
]

if __name__ == "__main__":
    try:    
        # Initialize robot
        robot = rb.Robot()
        robot.startup()
        controller = NormalizedVelocityControl(robot)
        anim = AnimatedStateControl(robot, drop_seq)
        anim.reverse = True
        
        while True:
            # Get command
            cmd = anim.get_command()
            print(f"Progress: {anim.get_progress():.2f}")

            # Send command
            cmd = hc.hybridize(cmd)
            controller.set_command(cmd)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.stop()
        if 'robot' in locals():
            robot.stop()
