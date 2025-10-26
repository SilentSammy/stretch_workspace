import sys
import os
# Add hybrid_control directory to path (assumes it's a sibling directory)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hybrid_control'))
import hybrid_control as hc
import state_control as sc
import normalized_velocity_control as nvc

import d405_helpers as dh
from aruco_detector import ArucoDetector
import cv2
import numpy as np

def get_frames():
    # Psuedo-static variable for realsense pipeline
    get_frames.pipeline = pipeline = getattr(get_frames, 'pipeline', None) or dh.start_d405(exposure='auto')[0]
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image

def get_norm_target_pos(rgb_frame, drawing_frame = None):
    # Pseudo-static variables for aruco detector and persistence
    get_norm_target_pos.aruco_det = aruco_det = getattr(get_norm_target_pos, 'aruco_det', None) or cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000), cv2.aruco.DetectorParameters())
    get_norm_target_pos.last_target = getattr(get_norm_target_pos, 'last_target', None)
    get_norm_target_pos.persistence_count = getattr(get_norm_target_pos, 'persistence_count', 0)

    corners, ids, _ = aruco_det.detectMarkers(rgb_frame)
    
    # Try to detect marker
    current_target = None
    if ids is not None:
        markers = zip(corners, ids.flatten())
        markers = [(c, i) for c, i in markers if i == 202]
        if len(markers) > 0:
            marker = markers[0]
            pos = np.mean(marker[0][0], axis=0)
            
            # Draw marker visualization
            if drawing_frame is not None:
                cv2.aruco.drawDetectedMarkers(drawing_frame, [marker[0]], np.array([[marker[1]]]))
                
                # Draw line from marker centroid to image center
                center = (rgb_frame.shape[1]//2, rgb_frame.shape[0]//2)
                marker_pos = (int(pos[0]), int(pos[1]))
                cv2.line(drawing_frame, marker_pos, center, (0, 255, 255), 2)  # Yellow line
                cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)  # Red dot at center
            
            # Calculate normalized position
            current_target = ((pos[0] - rgb_frame.shape[1]/2) / (rgb_frame.shape[1]/2),
                            (pos[1] - rgb_frame.shape[0]/2) / (rgb_frame.shape[0]/2))
    
    # Handle persistence logic
    if current_target is not None:
        # Fresh detection - update and reset persistence
        get_norm_target_pos.last_target = current_target
        get_norm_target_pos.persistence_count = 5  # Persist for 5 calls
        return current_target
    elif get_norm_target_pos.persistence_count > 0:
        # Use persisted target
        get_norm_target_pos.persistence_count -= 1
        return get_norm_target_pos.last_target
    else:
        # No target and persistence expired
        return None

def follow_target_w_wrist(norm_target_pos):
    import math
    
    # Pseudo-static state controller for roll correction
    follow_target_w_wrist.roll_controller = getattr(follow_target_w_wrist, 'roll_controller', None) or sc.StateControl(robot, {"wrist_roll": 0.0})
    
    # Start with roll correction command
    cmd = follow_target_w_wrist.roll_controller.get_command()
    
    # Get current wrist roll angle
    wrist_roll = robot.end_of_arm.motors['wrist_roll'].status['pos']
    
    # Raw visual error (negative because we want to move toward target)
    visual_error_x = -norm_target_pos[0]  # Camera x -> yaw correction
    visual_error_y = -norm_target_pos[1]  # Camera y -> pitch correction
    
    # Transform visual errors based on wrist roll using rotation matrix
    # When roll=0: x->yaw, y->pitch
    # When roll=90°: x->pitch, y->-yaw (rotated 90° CCW)
    cos_roll = math.cos(wrist_roll)
    sin_roll = math.sin(wrist_roll)
    
    # Apply 2D rotation matrix to transform camera coordinates to joint coordinates
    yaw_correction = cos_roll * visual_error_x - sin_roll * visual_error_y
    pitch_correction = sin_roll * visual_error_x + cos_roll * visual_error_y
    
    Kp = 1.0
    # Add visual servoing commands to roll correction
    cmd.update({
        "wrist_pitch_up": max(-1.0, min(1.0, pitch_correction * Kp)),
        "wrist_yaw_counterclockwise": max(-1.0, min(1.0, yaw_correction * Kp)),
    })
    
    return cmd

def face_target():
    import math
    
    # Get current wrist yaw position
    current_wrist_yaw = robot.end_of_arm.motors['wrist_yaw'].status['pos']
    desired_wrist_yaw = math.radians(90)  # 90 degrees in radians
    
    # Calculate error (how much the wrist has rotated from stowed)
    yaw_error = current_wrist_yaw - desired_wrist_yaw
    
    # Skip if within tolerance
    tolerance = math.radians(5)  # 5 degree tolerance
    if abs(yaw_error) <= tolerance:
        return {}
    
    # Use proportional control to rotate platform opposite to wrist rotation
    # If wrist yawed left (positive), platform should turn right (negative) to compensate
    Kp = 1.0
    platform_velocity = yaw_error * Kp  # Negative because platform turns opposite to wrist
    
    # Clamp velocity
    platform_velocity = max(-1.0, min(1.0, platform_velocity))
    
    return {"base_counterclockwise": platform_velocity}

try:
    controller = hc.get_controller()
    robot = controller.robot
    stow_controller = sc.StateControl(robot, sc.stowed_state)
    while True:
        rgb_image, depth_image = get_frames()
        drawing_frame = np.copy(rgb_image)
        norm_target_pos = get_norm_target_pos(rgb_image, drawing_frame)

        cv2.imshow('Wrist Cam', drawing_frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

        # Get commands
        stow_cmd = stow_controller.get_command()
        wrist_cmd = {}
        platform_cmd = {}
        if norm_target_pos is not None:
            wrist_cmd = follow_target_w_wrist(norm_target_pos)
            platform_cmd = face_target()

        # Split visual commands: yaw (priority) vs other axes
        wrist_yaw_cmd = {k: v for k, v in wrist_cmd.items() if k == "wrist_yaw_counterclockwise"}
        
        # Layer commands: lowest to highest priority
        cmd = nvc.zero_vel.copy()         # Baseline: all joints = 0
        cmd = hc.merge_override(platform_cmd, cmd)     # Platform rotation to face target
        cmd = hc.merge_override(wrist_cmd, cmd)        # Visual servoing (roll/pitch)
        cmd = hc.merge_override(stow_cmd, cmd)         # Stow dominates roll/pitch
        cmd = hc.merge_override(wrist_yaw_cmd, cmd)    # Visual yaw tracking

        cmd = hc.hybridize(cmd) # Human override
        
        print(f"Visual: {wrist_cmd} | Stow: {stow_cmd} | Platform: {platform_cmd} | Final: {cmd}")
        controller.set_command(cmd)
finally:
    cv2.destroyAllWindows()
    get_frames.pipeline.stop()
