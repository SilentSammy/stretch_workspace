#!/usr/bin/env python3

import stretch_body.robot as rb
import time

def get_all_joint_positions():
    """Get current positions of all joints on the Stretch robot"""
    
    try:
        # Initialize and startup the robot
        robot = rb.Robot()
        robot.startup()
        
        print("Getting current joint positions...")
        print("=" * 50)
        
        # Get basic joint positions
        arm_pos = robot.arm.status['pos']
        lift_pos = robot.lift.status['pos']
        
        # Get wheel positions
        left_wheel_pos = robot.base.left_wheel.status['pos']
        right_wheel_pos = robot.base.right_wheel.status['pos']
        
        # Get wrist joint positions
        wrist_roll_pos = robot.end_of_arm.motors['wrist_roll'].status['pos']
        wrist_pitch_pos = robot.end_of_arm.motors['wrist_pitch'].status['pos']
        wrist_yaw_pos = robot.end_of_arm.motors['wrist_yaw'].status['pos']
        
        # Get head joint positions
        head_pan_pos = robot.head.status['head_pan']['pos']
        head_tilt_pos = robot.head.status['head_tilt']['pos']
        
        # Get gripper position
        gripper_pos = robot.end_of_arm.motors['stretch_gripper'].status['pos']
        gripper_pos_pct = robot.end_of_arm.motors['stretch_gripper'].status['pos_pct']
        
        # Get base odometry (position in space)
        base_x = robot.base.status['x']
        base_y = robot.base.status['y']
        base_theta = robot.base.status['theta']
        
        # Print all positions
        print(f"Arm position:           {arm_pos:.4f} m")
        print(f"Lift position:          {lift_pos:.4f} m")
        print("")
        print(f"Left wheel position:    {left_wheel_pos:.4f} rad")
        print(f"Right wheel position:   {right_wheel_pos:.4f} rad")
        print("")
        print(f"Wrist roll position:    {wrist_roll_pos:.4f} rad")
        print(f"Wrist pitch position:   {wrist_pitch_pos:.4f} rad")
        print(f"Wrist yaw position:     {wrist_yaw_pos:.4f} rad")
        print("")
        print(f"Head pan position:      {head_pan_pos:.4f} rad")
        print(f"Head tilt position:     {head_tilt_pos:.4f} rad")
        print("")
        print(f"Gripper position:       {gripper_pos:.4f} rad")
        print(f"Gripper position (%):   {gripper_pos_pct:.1f}%")
        print("")
        print(f"Base odometry X:        {base_x:.4f} m")
        print(f"Base odometry Y:        {base_y:.4f} m")
        print(f"Base odometry Theta:    {base_theta:.4f} rad")
        
        # Stop the robot
        robot.stop()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'robot' in locals():
            robot.stop()

if __name__ == "__main__":
    get_all_joint_positions()