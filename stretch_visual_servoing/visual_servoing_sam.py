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
import math
import time

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
            
            # Calculate normalized position
            current_target = ((pos[0] - rgb_frame.shape[1]/2) / (rgb_frame.shape[1]/2),
                            (pos[1] - rgb_frame.shape[0]/2) / (rgb_frame.shape[0]/2))
    
    # Handle persistence logic
    if current_target is not None:
        # Fresh detection - update and reset persistence
        get_norm_target_pos.last_target = current_target
        get_norm_target_pos.persistence_count = 10  # Persist for 5 calls
        return current_target
    elif get_norm_target_pos.persistence_count > 0:
        # Use persisted target
        get_norm_target_pos.persistence_count -= 1
        return get_norm_target_pos.last_target
    else:
        # No target and persistence expired
        return None

def follow_target_w_wrist(norm_target_pos, drawing_frame=None, target_offset = (0, 0)):
    import math
    
    # Pseudo-static state controller for roll correction
    follow_target_w_wrist.roll_controller = getattr(follow_target_w_wrist, 'roll_controller', None) or sc.StateControl(robot, {"wrist_roll": 0.0})
    
    # Start with roll correction command
    cmd = follow_target_w_wrist.roll_controller.get_command()
    
    # Get current wrist roll angle
    wrist_roll = robot.end_of_arm.motors['wrist_roll'].status['pos']
    
    # Raw visual error (negative because we want to move toward target)
    # Apply vertical offset: target (0, -0.1) instead of center (0, 0)
    target_offset_x, target_offset_y = target_offset
    visual_error_x = -(norm_target_pos[0] + target_offset_x)  # Camera x -> yaw correction
    visual_error_y = -(norm_target_pos[1] + target_offset_y)  # Camera y -> pitch correction
    
    # Transform visual errors based on wrist roll using rotation matrix
    # When roll=0: x->yaw, y->pitch
    # When roll=90°: x->pitch, y->-yaw (rotated 90° CCW)
    cos_roll = math.cos(wrist_roll)
    sin_roll = math.sin(wrist_roll)
    
    # Apply 2D rotation matrix to transform camera coordinates to joint coordinates
    yaw_correction = cos_roll * visual_error_x - sin_roll * visual_error_y
    pitch_correction = sin_roll * visual_error_x + cos_roll * visual_error_y
    
    Kp = 0.5
    # Add visual servoing commands to roll correction
    cmd.update({
        "wrist_pitch_up": max(-1.0, min(1.0, pitch_correction * Kp)),
        "wrist_yaw_counterclockwise": max(-1.0, min(1.0, yaw_correction * Kp)),
    })
    
    # Draw line from marker to target setpoint if drawing_frame is provided
    if drawing_frame is not None:
        height, width = drawing_frame.shape[:2]
        # Convert normalized positions to pixel coordinates
        marker_x = int((norm_target_pos[0] + 1.0) * width / 2.0)
        marker_y = int((norm_target_pos[1] + 1.0) * height / 2.0)
        target_x = int((-target_offset_x + 1.0) * width / 2.0)
        target_y = int((-target_offset_y + 1.0) * height / 2.0)  # Flip sign for display
        
        # Draw crosshair at actual image center
        center_x = width // 2
        center_y = height // 2
        crosshair_size = 10
        cv2.line(drawing_frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (255, 255, 255), 1)  # White horizontal line
        cv2.line(drawing_frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (255, 255, 255), 1)  # White vertical line
        
        # Draw line from marker to target setpoint
        cv2.line(drawing_frame, (marker_x, marker_y), (target_x, target_y), (0, 255, 255), 2)  # Yellow line
        cv2.circle(drawing_frame, (target_x, target_y), 5, (0, 0, 255), -1)  # Red dot at target
    
    return cmd

def face_target(target_yaw_degrees=90):
    import math
    
    # Get current wrist yaw position
    current_wrist_yaw = robot.end_of_arm.motors['wrist_yaw'].status['pos']
    desired_wrist_yaw = math.radians(target_yaw_degrees)  # Convert degrees to radians
    
    # Calculate error (how much the wrist has rotated from target)
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

def present_flank():
    return face_target(target_yaw_degrees=0)

def get_target_distance(rgb_frame, depth_frame, norm_target_pos):
    # Pseudo-static variables for distance history
    get_target_distance.history = getattr(get_target_distance, 'history', [])
    history_size = 5
    
    # Convert normalized position back to pixel coordinates
    height, width = rgb_frame.shape[:2]
    pixel_x = int((norm_target_pos[0] + 1.0) * width / 2.0)
    pixel_y = int((norm_target_pos[1] + 1.0) * height / 2.0)
    
    # Clamp to image bounds
    pixel_x = max(0, min(width - 1, pixel_x))
    pixel_y = max(0, min(height - 1, pixel_y))
    
    # RealSense D405 depth scale (converts raw depth units to meters)
    depth_scale = 9.999999747378752e-05  # ~0.0001 meters per unit
    
    # Get raw depth value at target position
    raw_depth = depth_frame[pixel_y, pixel_x]
    
    # Convert to actual distance in meters
    distance_m = raw_depth * depth_scale
    
    # Sample a small region around the target for more robust measurement
    sample_size = 5
    y_start = max(0, pixel_y - sample_size//2)
    y_end = min(height, pixel_y + sample_size//2 + 1)
    x_start = max(0, pixel_x - sample_size//2)
    x_end = min(width, pixel_x + sample_size//2 + 1)
    
    depth_region = depth_frame[y_start:y_end, x_start:x_end]
    
    # Filter out zero/invalid depths
    valid_depths = depth_region[depth_region > 0]
    
    if len(valid_depths) > 0:
        # Use median for robustness
        median_raw_depth = np.median(valid_depths)
        median_distance_m = median_raw_depth * depth_scale
        
        # Add to history and maintain size
        get_target_distance.history.append(median_distance_m)
        if len(get_target_distance.history) > history_size:
            get_target_distance.history.pop(0)  # Remove oldest
        
        # Return historical average
        return sum(get_target_distance.history) / len(get_target_distance.history)
    else:
        return None

def move_to_target(target_dist):
    import math
    
    # Tuning parameters
    target_distance = 0.60  # setpoint in cm
    max_align_error = math.radians(20)  # 0% authority at this angle (100% at 0°)
    max_forward_speed = 1.0  # Maximum forward velocity (increased for more aggressive response)
    distance_kp = 5.0  # Proportional gain for distance control (doubled for more aggressive response)
    
    # Always-active proportional distance controller
    distance_error = target_dist - target_distance
    distance_velocity = distance_kp * distance_error
    
    # Get current wrist yaw alignment error
    current_wrist_yaw = robot.end_of_arm.motors['wrist_yaw'].status['pos']
    desired_wrist_yaw = math.radians(90)  # 90 degrees reference
    yaw_error = abs(current_wrist_yaw - desired_wrist_yaw)
    
    # Linear authority scaling: 1.0 at 0°, 0.0 at 15°
    authority = max(0.0, 1.0 - (yaw_error / max_align_error))
    
    # Scale forward velocity by alignment authority
    forward_velocity = distance_velocity * authority
    
    # Clamp velocity
    forward_velocity = max(-max_forward_speed, min(max_forward_speed, forward_velocity))
    
    return {"base_forward": forward_velocity}

def inside_grasp_window(target_dist):
    # Pseudo-static variable to track grasp window state
    inside_grasp_window.in_window = getattr(inside_grasp_window, 'in_window', True)
    
    # Base hysteresis parameters
    enter_min, enter_max = 0.58, 0.62
    base_exit_min, exit_max = 0.40, 0.80
    
    if target_dist is None:
        return inside_grasp_window.in_window
    
    # Get current arm extension and calculate extension-aware exit limit
    arm_extension = robot.arm.status['pos']  # Current arm position in meters
    effective_exit_min = base_exit_min - arm_extension  # Lower limit decreases as arm extends
    effective_exit_min = max(0.10, effective_exit_min)  # Never go below 10cm for safety
    
    if not inside_grasp_window.in_window:
        # Outside window - check if we should enter
        if enter_min <= target_dist <= enter_max:
            inside_grasp_window.in_window = True
    else:
        # Inside window - check if we should exit (with extension-aware lower limit)
        if target_dist < effective_exit_min or target_dist > exit_max:
            inside_grasp_window.in_window = False
    
    return inside_grasp_window.in_window

def get_reach_authority(current_wrist_yaw, target_yaw=0.0, start_angle_deg=30, complete_angle_deg=15):
    """Calculate progressive authority handover based on wrist yaw alignment"""
    import math
    
    yaw_error = abs(current_wrist_yaw - target_yaw)  # Distance from target position
    
    # Convert angles to radians
    handover_start = math.radians(start_angle_deg)
    handover_complete = math.radians(complete_angle_deg)
    
    # Calculate mix ratio: 0.0 at start_angle (all stow), 1.0 at complete_angle (all visual)
    if yaw_error <= handover_complete:
        mix_ratio = 1.0  # Full visual control
    else:
        # Linear interpolation between start and complete angles
        progress = (handover_start - yaw_error) / (handover_start - handover_complete)
        mix_ratio = max(0.0, min(1.0, progress))
    
    return mix_ratio

def reach_to_target(target_dist):
    # Target distance for reaching
    target_distance = 0.15  # meters
    
    # Proportional controller for arm extension
    distance_error = target_dist - target_distance
    Kp_arm = 2.0  # Proportional gain for arm
    
    # Calculate arm extension velocity (negative error means extend arm)
    arm_velocity = distance_error * Kp_arm
    
    # Pitch correction via lift adjustment
    current_wrist_pitch = robot.end_of_arm.motors['wrist_pitch'].status['pos']
    desired_wrist_pitch = math.radians(-15)  # target pitch
    
    # Calculate pitch error
    pitch_error = current_wrist_pitch - desired_wrist_pitch
    
    # Skip lift adjustment if within tolerance
    tolerance = math.radians(2)  # degree tolerance
    if abs(pitch_error) <= tolerance:
        lift_velocity = 0.0
    else:
        # Use proportional control for lift adjustment
        # If pitch is too low (negative error), lower lift (negative velocity)
        # If pitch is too high (positive error), raise lift (positive velocity)
        Kp_lift = 1.0
        lift_velocity = pitch_error * Kp_lift
        
        # Clamp lift velocity
        lift_velocity = max(-1.0, min(1.0, lift_velocity))
    
    # Clamp arm velocity
    arm_velocity = max(-1.0, min(1.0, arm_velocity))
    
    return {
        "arm_out": arm_velocity,
        "lift_up": lift_velocity
    }

def is_graspable(norm_target_pos, target_dist):
    # Pseudo-static variables for stability tracking
    is_graspable.last_pos = getattr(is_graspable, 'last_pos', None)
    is_graspable.last_dist = getattr(is_graspable, 'last_dist', None)
    is_graspable.stable_start_time = getattr(is_graspable, 'stable_start_time', None)
    
    # Stability thresholds
    pos_threshold = 0.1  # Position change threshold (relaxed)
    dist_threshold = 0.5  # Distance change threshold (meters)
    target_distance_min = 0.12  # Minimum target distance (12cm)
    target_distance_max = 0.16  # Maximum target distance (16cm)
    stability_duration = 3.0  # Required stable time (seconds)
    
    # Check if target distance is in acceptable range (12-16cm)
    if target_dist is None or target_dist < target_distance_min or target_dist > target_distance_max:
        # Reset stability tracking if distance is out of range
        is_graspable.stable_start_time = None
        is_graspable.last_pos = norm_target_pos
        is_graspable.last_dist = target_dist
        return False
    
    # Check stability if we have previous measurements
    if is_graspable.last_pos is not None and is_graspable.last_dist is not None:
        # Calculate position change (Euclidean distance)
        pos_change = math.sqrt((norm_target_pos[0] - is_graspable.last_pos[0])**2 + 
                              (norm_target_pos[1] - is_graspable.last_pos[1])**2)
        dist_change = abs(target_dist - is_graspable.last_dist)
        
        # Check if currently stable
        is_currently_stable = (pos_change <= pos_threshold and dist_change <= dist_threshold)
        
        if is_currently_stable:
            # Start timing if this is the first stable frame
            if is_graspable.stable_start_time is None:
                is_graspable.stable_start_time = time.time()
            
            # Check if we've been stable long enough
            stable_duration = time.time() - is_graspable.stable_start_time
            if stable_duration >= stability_duration:
                return True  # Ready to grasp!
        else:
            # Reset stability timer if not stable
            is_graspable.stable_start_time = None
    
    # Update tracking variables
    is_graspable.last_pos = norm_target_pos
    is_graspable.last_dist = target_dist
    
    return False

def is_grasped(graspable):
    pass

try:
    controller = hc.get_controller()
    robot = controller.robot
    stow_controller = sc.StateControl(robot, sc.stowed_state)
    grasp_controller = sc.StateControl(robot, {"gripper": math.radians(90)})
    while True:
        # Get sensor data
        rgb_image, depth_image = get_frames()
        drawing_frame = np.copy(rgb_image)

        # Initialize commands
        stow_cmd = stow_controller.get_command()
        cmd = nvc.zero_vel.copy()         # Baseline: all joints = 0
        cmd = hc.merge_override(stow_cmd, cmd)
        platform_cmd = {}
        wrist_yaw_cmd = {}
        grasp_cmd = {}

        # Get target data
        norm_target_pos = get_norm_target_pos(rgb_image, drawing_frame)
        dist = get_target_distance(rgb_image, depth_image, norm_target_pos) if norm_target_pos is not None else None
        graspable = is_graspable(norm_target_pos, dist)

        if norm_target_pos is not None:
            wrist_cmd = follow_target_w_wrist(norm_target_pos)      
            if dist is not None:
                in_grasp_window = inside_grasp_window(dist)
                print(f"Target distance: {dist:.2f}m | Grasp window: {in_grasp_window}")
                    
                if in_grasp_window: # Grasping behavior: present flank to target
                    platform_cmd = present_flank()  # Orient side toward target (0° wrist yaw)

                    # Calculate progressive authority handover based on wrist yaw alignment
                    current_wrist_yaw = robot.end_of_arm.motors['wrist_yaw'].status['pos']
                    reach_auth = get_reach_authority(current_wrist_yaw, target_yaw=0.0, start_angle_deg=30, complete_angle_deg=15)                    # Mix stow and visual commands for reaching behavior
                    reach_cmd = reach_to_target(dist)

                    offset = (-0.1 * reach_auth, -0.4 * reach_auth)
                    wrist_cmd = follow_target_w_wrist(norm_target_pos, drawing_frame, offset)
                    cmd = hc.merge_override(hc.merge_mix(wrist_cmd, stow_cmd, reach_auth), cmd)
                    cmd = hc.merge_mix(reach_cmd, cmd, reach_auth)
                    yaw_error = abs(current_wrist_yaw - 0.0)  # For debug display
                    print(f"Wrist yaw: {math.degrees(yaw_error):.1f}° | Authority: {reach_auth:.2f}" + (f" | Graspable!" if graspable else ""))
                    if graspable:
                        print("Grasping target!")
                        grasp_cmd = grasp_controller.get_command()
                        cmd = hc.merge_override(grasp_cmd, cmd)         # Grasping
                else: # Approach behavior: face and move to target
                    platform_cmd = face_target()  # Start with rotation commands
                    platform_cmd.update(move_to_target(dist))  # Merge in forward motion
            wrist_yaw_cmd = {k: v for k, v in wrist_cmd.items() if k == "wrist_yaw_counterclockwise"}
        
        # Layer commands: lowest to highest priority
        cmd = hc.merge_override(wrist_yaw_cmd, cmd)    # Visual yaw tracking
        cmd = hc.merge_override(platform_cmd, cmd)     # Platform rotation + forward motion
        cmd = hc.hybridize(cmd) # Human override
        
        # print(f"Visual: {wrist_cmd} | Stow: {stow_cmd} | Platform: {platform_cmd} | Final: {cmd}")
        controller.set_command(cmd)

        # Display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Wrist Cam', drawing_frame)
        cv2.imshow('Depth View', depth_colormap)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
finally:
    cv2.destroyAllWindows()
    get_frames.pipeline.stop()
