#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gamepad_interface as gp
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
import stretch_body.robot as rb
import time
from dataclasses import dataclass

@dataclass
class JointSettings:
    gamepad_input: str
    joint_command: str
    scale_factor: float
    gamepad_input_negative: str = ""  # Optional secondary input for negative direction

# Joint mapping configuration
joint_settings = [
    # Base control (left stick)
    JointSettings("left_stick_y", "base_forward", 1.0),
    JointSettings("left_stick_x", "base_counterclockwise", -1.0),
    
    # Arm control (right stick)
    JointSettings("right_stick_y", "lift_up", 1.0),         # Right stick Y -> Arm up/down (lift)
    JointSettings("right_stick_x", "arm_out", 1.0),         # Right stick X -> Arm extend/retract
    
    # Wrist control (D-pad and bumpers) - using dual digital inputs
    JointSettings("right_pad_pressed", "wrist_roll_counterclockwise", 1.0, "left_pad_pressed"),
    JointSettings("top_pad_pressed", "wrist_pitch_up", 1.0, "bottom_pad_pressed"),
    JointSettings("left_shoulder_button_pressed", "wrist_yaw_counterclockwise", 1.0, "right_shoulder_button_pressed"),
    
    # Gripper control (face buttons) - using dual digital inputs
    JointSettings("right_button_pressed", "gripper_open", 1.0, "bottom_button_pressed"),
]

# Joints affected by manual mode (algorithm disabled for these joints when manual mode is on)
manual_mode_joints = [
    "wrist_yaw_counterclockwise",
    "wrist_roll_counterclockwise", 
    "wrist_pitch_up",
    # Uncomment to disable base movement in manual mode
    # "base_forward",
    # "base_counterclockwise",
]

print("Hybrid Joystick Control")
print("Y button: Toggle manual mode")

# Manual mode state
manual_mode = False
y_button_last_state = False

def get_gamepad_cmd():
    # Create base command (all zeros)
    cmd = zero_vel.copy()
    
    # Process each joint setting
    for joint in joint_settings:
        # Get primary input value
        input_value = gp.get_axis(joint.gamepad_input)
        
        # If there's a secondary negative input, combine them
        if joint.gamepad_input_negative:
            negative_input_value = gp.get_axis(joint.gamepad_input_negative)
            # Primary input = positive direction, secondary = negative direction
            combined_value = input_value - negative_input_value
            joint_velocity = combined_value * joint.scale_factor
        else:
            # Standard analog input
            joint_velocity = input_value * joint.scale_factor
        
        cmd[joint.joint_command] = joint_velocity
    
    return cmd

def display_status(clean_output=True):
    if clean_output:
        print("\033[H\033[J", end="")  # Clear screen and move cursor to top
        mode_text = "MANUAL MODE" if manual_mode else "HYBRID MODE"
        print(f"Hybrid Joystick Control - {mode_text}")
        print("Y button: Toggle manual mode")
        print("Active Mappings:")
    
    for joint in joint_settings:
        input_value = gp.get_axis(joint.gamepad_input)
        
        if joint.gamepad_input_negative:
            negative_input_value = gp.get_axis(joint.gamepad_input_negative)
            combined_value = input_value - negative_input_value
            joint_velocity = combined_value * joint.scale_factor
            print(f"  {joint.gamepad_input}(+) / {joint.gamepad_input_negative}(-): {combined_value:6.3f} -> {joint.joint_command}: {joint_velocity:6.3f}")
        else:
            joint_velocity = input_value * joint.scale_factor
            print(f"  {joint.gamepad_input}: {input_value:6.3f} -> {joint.joint_command}: {joint_velocity:6.3f}")

def get_algo_cmd():
    import math
    # Create sine waves for wrist yaw and roll
    t = time.time()
    yaw_frequency = 0.2   # Hz - slow oscillation
    roll_frequency = 0.15 # Hz - slightly different frequency for interesting motion
    amplitude = 0.5       # Max velocity (-0.5 to 0.5)
    
    cmd = zero_vel.copy()
    cmd['wrist_yaw_counterclockwise'] = amplitude * math.sin(2 * math.pi * yaw_frequency * t)
    cmd['wrist_roll_counterclockwise'] = amplitude * math.sin(2 * math.pi * roll_frequency * t + math.pi/4)  # Phase offset
    
    return cmd

def disable_joints(cmd_dict, joints_to_disable):
    cmd_dict = cmd_dict.copy()
    for joint in joints_to_disable:
        cmd_dict[joint] = 0.0
    return cmd_dict

def apply_manual_mode(cmd_dict):
    if manual_mode:
        cmd_dict = disable_joints(cmd_dict, manual_mode_joints)
    return cmd_dict

def merge_proportional(cmd_gamepad, cmd_algo):
    # User input scales from algo value toward their desired value
    cmd_final = {}
    
    # Handle all joints from both commands
    all_joints = set(cmd_gamepad.keys()) | set(cmd_algo.keys())
    
    for joint in all_joints:
        user_input = cmd_gamepad.get(joint, 0.0)  # Default to 0 if missing
        algo_input = cmd_algo.get(joint, 0.0)     # Default to 0 if missing
        
        if abs(user_input) < 0.05:  # No user input
            cmd_final[joint] = algo_input
        else:
            # User input interpolates between algo and desired value
            # abs(user_input) determines how much override (0 to 1)
            # sign(user_input) determines direction
            override_strength = abs(user_input)
            desired_value = 1.0 if user_input > 0 else -1.0
            cmd_final[joint] = (1 - override_strength) * algo_input + override_strength * desired_value
    
    return cmd_final

def hybridize(cmd_dict):
    global manual_mode, y_button_last_state
    """ Call this inside your algorithm loop to merge gamepad and algo commands """
    # Handle Y button toggle for manual mode
    y_button_current = gp.get_axis("top_button_pressed")
    
    if y_button_current and not y_button_last_state:  # Button just pressed
        manual_mode = not manual_mode
        print(f"\nMode switched to: {'MANUAL' if manual_mode else 'HYBRID'}")
    
    y_button_last_state = y_button_current
    
    # Get gamepad command
    user_cmd = get_gamepad_cmd()

    # Merge commands
    cmd = merge_proportional(user_cmd, cmd_dict)

    return cmd

if __name__ == "__main__":
    try:
        # Initialize robot
        robot = rb.Robot()
        robot.startup()

        # Create velocity controller
        controller = NormalizedVelocityControl(robot)

        while True:

            # Get algorithmic command
            algo_cmd = apply_manual_mode(get_algo_cmd())

            # Hybridize commands
            hybrid_cmd = hybridize(algo_cmd)

            # Send command
            controller.set_command(hybrid_cmd)

            # Display status
            display_status(clean_output=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.set_command(zero_vel)
        controller.stop()
        gp.stop_gamepad()
        robot.stop()
        print("Done!")
