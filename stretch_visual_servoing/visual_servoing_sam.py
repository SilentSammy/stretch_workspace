import sys
import os
# Add hybrid_control directory to path (assumes it's a sibling directory)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hybrid_control'))
import hybrid_control as hc
import state_control as sc
import normalized_velocity_control as nvc
import pyrealsense2 as rs
from target_finder import ArucoTargetFinder
from tennisfinder import TennisFinder

import d405_helpers as dh
from aruco_detector import ArucoDetector
import cv2
import numpy as np
import math
import time

def get_head_cam_frames():
    # Pseudo-static variable for realsense pipeline
    get_head_cam_frames.pipeline = getattr(get_head_cam_frames, 'pipeline', None)
    
    if get_head_cam_frames.pipeline is None:
        # Initialize pipeline for head camera (D435)
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams for D435 head camera
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        
        # Start pipeline
        pipeline.start(config)
        get_head_cam_frames.pipeline = pipeline
    
    frames = get_head_cam_frames.pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Rotate 90 degrees clockwise
    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    
    return color_image, depth_image

def get_wrist_cam_frames():
    # Psuedo-static variable for realsense pipeline
    get_wrist_cam_frames.pipeline = pipeline = getattr(get_wrist_cam_frames, 'pipeline', None) or dh.start_d405(exposure='auto')[0]
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image

def get_norm_target_pos(rgb_frame, drawing_frame=None):
    # Pseudo-static target finder instance
    # get_norm_target_pos.target_finder = getattr(get_norm_target_pos, 'target_finder', None) or ArucoTargetFinder(
    #     target_ids=202,
    #     aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000), 
    #     persistence_frames=10
    # )
    get_norm_target_pos.target_finder = getattr(get_norm_target_pos, 'target_finder', None) or TennisFinder()
    
    return get_norm_target_pos.target_finder.get_normalized_target_position(rgb_frame, drawing_frame)

def get_norm_dest_pos(rgb_frame, drawing_frame=None):
    # Pseudo-static destination finder instance  
    get_norm_dest_pos.destination_finder = getattr(get_norm_dest_pos, 'destination_finder', None) or ArucoTargetFinder(
        target_ids=12, 
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
        persistence_frames=5
    )
    return get_norm_dest_pos.destination_finder.get_normalized_target_position(rgb_frame, drawing_frame, False)

def follow_target_w_wrist(norm_target_pos, drawing_frame=None, target_offset = (0, 0)):
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

def follow_destination_w_head(norm_dest_pos, drawing_frame=None, target_offset=(0, 0)):
    # Pseudo-static state controller for head centering
    follow_destination_w_head.head_controller = getattr(follow_destination_w_head, 'head_controller', None) or sc.StateControl(robot, {"head_pan": 0.0, "head_tilt": 0.0})
    
    # Start with head centering command
    cmd = follow_destination_w_head.head_controller.get_command()
    
    # Raw visual error (negative because we want to move toward destination)
    target_offset_x, target_offset_y = target_offset
    visual_error_x = -(norm_dest_pos[0] + target_offset_x)  # Camera x -> pan correction
    visual_error_y = -(norm_dest_pos[1] + target_offset_y)  # Camera y -> tilt correction
    
    Kp = 0.2
    # Add visual servoing commands to head centering
    cmd.update({
        "head_tilt_up": max(-1.0, min(1.0, visual_error_y * Kp)),
        "head_pan_counterclockwise": max(-1.0, min(1.0, visual_error_x * Kp)),
    })
    
    # Draw line from destination marker to target setpoint if drawing_frame is provided
    if drawing_frame is not None:
        height, width = drawing_frame.shape[:2]
        # Convert normalized positions to pixel coordinates
        marker_x = int((norm_dest_pos[0] + 1.0) * width / 2.0)
        marker_y = int((norm_dest_pos[1] + 1.0) * height / 2.0)
        target_x = int((-target_offset_x + 1.0) * width / 2.0)
        target_y = int((-target_offset_y + 1.0) * height / 2.0)  # Flip sign for display
        
        # Draw crosshair at actual image center
        center_x = width // 2
        center_y = height // 2
        crosshair_size = 10
        cv2.line(drawing_frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (255, 255, 255), 1)  # White horizontal line
        cv2.line(drawing_frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (255, 255, 255), 1)  # White vertical line
        
        # Draw line from destination marker to target setpoint
        cv2.line(drawing_frame, (marker_x, marker_y), (target_x, target_y), (0, 255, 0), 2)  # Green line for destination
        cv2.circle(drawing_frame, (target_x, target_y), 5, (0, 0, 255), -1)  # Red dot at target
    
    return cmd

def present_flank():
    return face_target(target_yaw_degrees=0)

def face_target(target_yaw_degrees=90):
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

def present_flank_with_head():
    return face_head(target_pan_degrees=-90)

def face_head(target_pan_degrees=0):
    # Get current head pan position
    current_head_pan = robot.head.status['head_pan']['pos']
    desired_head_pan = math.radians(target_pan_degrees)  # Convert degrees to radians
    
    # Calculate error (how much the head has panned from target)
    pan_error = current_head_pan - desired_head_pan
    
    # Skip if within tolerance
    tolerance = math.radians(5)  # 5 degree tolerance
    if abs(pan_error) <= tolerance:
        return {}
    
    # Use proportional control to rotate platform opposite to head pan
    # If head panned left (positive), platform should turn right (negative) to compensate
    Kp = 1.0
    platform_velocity = pan_error * Kp  # Negative because platform turns opposite to head pan
    
    # Clamp velocity
    platform_velocity = max(-1.0, min(1.0, platform_velocity))
    
    return {"base_counterclockwise": platform_velocity}

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

def face_destination(norm_dest_pos):
    """Rotate base to face destination marker using head pan angle"""
    import math
    
    # Get current head pan position
    current_head_pan = robot.head.status['head_pan']['pos']
    
    # If destination is centered in head camera, we should be facing it
    # If destination is to the left (positive x), we need to turn left (positive base rotation)
    # If destination is to the right (negative x), we need to turn right (negative base rotation)
    
    # Use head pan angle as indication of where destination is relative to robot
    # Positive head pan = destination is to the left, so turn base left
    # Negative head pan = destination is to the right, so turn base right
    
    # Skip rotation if head is centered (destination is in front)
    tolerance = math.radians(10)  # 10 degree tolerance
    if abs(current_head_pan) <= tolerance:
        return {}
    
    # Use proportional control to rotate base in same direction as head pan
    Kp = 2.0  # Lower gain for smoother rotation
    platform_velocity = current_head_pan * Kp
    
    # Clamp velocity
    platform_velocity = max(-1.0, min(1.0, platform_velocity))
    
    return {"base_counterclockwise": platform_velocity}

def move_to_dest(dest_dist):
    """Move forward/backward to reach destination at optimal distance"""
    import math
    
    # Tuning parameters for destination approach
    target_distance = 1.0  # setpoint in meters
    max_forward_speed = 1.0  # Maximum forward velocity
    distance_kp = 5.0  # Proportional gain for distance control
    
    # Check if destination distance is available
    if dest_dist is None:
        return {}
    
    # Proportional distance controller
    distance_error = dest_dist - target_distance
    distance_velocity = distance_kp * distance_error
    
    # Get current head pan alignment (destination should be centered in front)
    current_head_pan = robot.head.status['head_pan']['pos']
    max_align_error = math.radians(20)  # 0% authority at this angle (100% at 0°)
    
    # Linear authority scaling: 1.0 at 0°, 0.0 at 20°
    authority = max(0.0, 1.0 - (abs(current_head_pan) / max_align_error))
    
    # Scale forward velocity by alignment authority
    forward_velocity = distance_velocity * authority
    
    # Clamp velocity
    forward_velocity = max(-max_forward_speed, min(max_forward_speed, forward_velocity))
    
    return {"base_forward": forward_velocity}

def get_distance_generic(rgb_frame, depth_frame, norm_pos, depth_scale):
    """Pure distance measurement function without any history management"""
    if norm_pos is None:
        return None
        
    # Convert normalized position back to pixel coordinates
    height, width = rgb_frame.shape[:2]
    pixel_x = int((norm_pos[0] + 1.0) * width / 2.0)
    pixel_y = int((norm_pos[1] + 1.0) * height / 2.0)
    
    # Clamp to image bounds
    pixel_x = max(0, min(width - 1, pixel_x))
    pixel_y = max(0, min(height - 1, pixel_y))
    
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
        return median_raw_depth * depth_scale
    else:
        return None

def get_wrist_distance(rgb_frame, depth_frame, norm_pos):
    """Get distance from wrist camera with separate history tracking"""
    get_wrist_distance.history = getattr(get_wrist_distance, 'history', [])
    history_size = 5
    
    # Get current distance measurement
    current_distance = get_distance_generic(rgb_frame, depth_frame, norm_pos, 1e-04)
    
    if current_distance is not None:
        # Add to history and maintain size
        get_wrist_distance.history.append(current_distance)
        if len(get_wrist_distance.history) > history_size:
            get_wrist_distance.history.pop(0)  # Remove oldest
        
        # Return historical average
        return sum(get_wrist_distance.history) / len(get_wrist_distance.history)
    else:
        return None

def get_head_distance(rgb_frame, depth_frame, norm_pos):
    """Get distance from head camera with separate history tracking"""
    get_head_distance.history = getattr(get_head_distance, 'history', [])
    history_size = 5
    
    # Get current distance measurement
    current_distance = get_distance_generic(rgb_frame, depth_frame, norm_pos, 1e-03)
    
    if current_distance is not None:
        # Add to history and maintain size
        get_head_distance.history.append(current_distance)
        if len(get_head_distance.history) > history_size:
            get_head_distance.history.pop(0)  # Remove oldest
        
        # Return historical average
        return sum(get_head_distance.history) / len(get_head_distance.history)
    else:
        return None

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

def inside_dest_window(dest_dist):
    # Pseudo-static variable to track destination window state
    inside_dest_window.in_window = getattr(inside_dest_window, 'in_window', True)
    
    # Base hysteresis parameters for destination approach
    enter_min, enter_max = 0.95, 1.05
    base_exit_min, exit_max = 0.90, 1.10
    
    if dest_dist is None:
        return inside_dest_window.in_window
    
    # Get current arm extension and calculate extension-aware exit limit
    arm_extension = robot.arm.status['pos']  # Current arm position in meters
    effective_exit_min = base_exit_min - arm_extension  # Lower limit decreases as arm extends
    effective_exit_min = max(0.50, effective_exit_min)  # Never go below 50cm for safety
    
    if not inside_dest_window.in_window:
        # Outside window - check if we should enter
        if enter_min <= dest_dist <= enter_max:
            inside_dest_window.in_window = True
    else:
        # Inside window - check if we should exit (with extension-aware lower limit)
        if dest_dist < effective_exit_min or dest_dist > exit_max:
            inside_dest_window.in_window = False
    
    return inside_dest_window.in_window

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

def reach_for_target(target_dist):
    # Target distance for reaching
    distance_setpoint = 0.15  # meters
    
    # Proportional controller for arm extension
    distance_error = target_dist - distance_setpoint
    Kp_arm = 1.0  # Proportional gain for arm
    
    # Calculate arm extension velocity (negative error means extend arm)
    arm_velocity = distance_error * Kp_arm
    
    # Calculate lift authority based on target distance
    # 0.0 authority at 30cm, 1.0 authority at 20cm
    lift_auth_start_dist = 0.50  # no lift authority
    lift_auth_complete_dist = 0.25  # full lift authority
    
    # Linear interpolation with clamping
    progress = (lift_auth_start_dist - target_dist) / (lift_auth_start_dist - lift_auth_complete_dist)
    lift_auth = max(0.0, min(1.0, progress))
    
    # Pitch correction via lift adjustment (only when lift authority is active)
    current_wrist_pitch = robot.end_of_arm.motors['wrist_pitch'].status['pos']
    desired_wrist_pitch = math.radians(-15)  # target pitch
    
    # Calculate pitch error
    pitch_error = current_wrist_pitch - desired_wrist_pitch
    
    # Skip lift adjustment if within tolerance or no authority
    tolerance = math.radians(2)  # degree tolerance
    if abs(pitch_error) <= tolerance or lift_auth == 0.0:
        lift_velocity = 0.0
    else:
        # Use proportional control for lift adjustment scaled by authority
        # If pitch is too low (negative error), lower lift (negative velocity)
        # If pitch is too high (positive error), raise lift (positive velocity)
        Kp_lift = 1.0
        lift_velocity = pitch_error * Kp_lift * lift_auth
        
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
    # Bypass for testing
    # return True

    # Pseudo-static history of graspable states for noise smoothing
    is_grasped.graspable_hist = getattr(is_grasped, 'graspable_hist', [])
    history_size = 10  # Keep last 10 graspable states
    
    # Add current graspable state to history
    is_grasped.graspable_hist.append(graspable)
    if len(is_grasped.graspable_hist) > history_size:
        is_grasped.graspable_hist.pop(0)  # Remove oldest
    
    # Consider graspable if ANY recent state was True (noise smoothing)
    smoothed_graspable = any(is_grasped.graspable_hist)
    
    # Get current gripper position in radians
    current_gripper_pos = robot.end_of_arm.motors['stretch_gripper'].status['pos']
    
    # Simple check: object is graspable and gripper is at expected starting position
    expected_gripper_pos = math.radians(90)  # 90 degrees
    tolerance = math.radians(15)  # degrees
    
    gripper_at_position = abs(current_gripper_pos - expected_gripper_pos) <= tolerance
    
    return smoothed_graspable and gripper_at_position

def get_interpolated_authority(current_value, zero_point, one_point):
    if zero_point > one_point:
        progress = (zero_point - current_value) / (zero_point - one_point)
    else:
        progress = (current_value - zero_point) / (one_point - zero_point)
    return max(0.0, min(1.0, progress))

try:
    controller = hc.get_controller()
    robot = controller.robot
    stow_controller = sc.StateControl(robot, sc.stowed_state)
    grasp_controller = sc.StateControl(robot, {"gripper": math.radians(90)})
    carry_controller = sc.StateControl(robot, sc.carry_state)
    raise_controller = sc.StateControl(robot, sc.raise_state)
    raise_controller
    zero_vel = nvc.zero_vel.copy()
    scan_cmd = {"base_counterclockwise": -0.75}

    extend_controller = sc.StateControl(robot, sc.extend_state)
    drop_controller = sc.StateControl(robot, sc.drop_state)

    at_dropoff_position = False
    while True:
        # Get sensor data
        wrist_rgb, wrist_depth = get_wrist_cam_frames()
        head_rgb, head_depth = get_head_cam_frames()
        wrist_drawing = np.copy(wrist_rgb)
        head_drawing = np.copy(head_rgb)

        # Initialize commands
        cmd = zero_vel
        platform_cmd = {}
        stow_cmd = stow_controller.get_command()

        # Get target data
        norm_target_pos = get_norm_target_pos(wrist_rgb, wrist_drawing)
        dist = get_wrist_distance(wrist_rgb, wrist_depth, norm_target_pos)

        # Get destination data
        norm_dest_pos = get_norm_dest_pos(head_rgb, head_drawing)
        dest_dist = get_head_distance(head_rgb, head_depth, norm_dest_pos)

        graspable = is_graspable(norm_target_pos, dist)
        grasped = is_grasped(graspable)

        if not at_dropoff_position: # Grab the object and take to drop-off
            if not grasped: # Grab the object
                cmd = hc.merge_override(stow_cmd, cmd)   # Stow
                if dist is not None: # If target is visible
                    wrist_cmd = follow_target_w_wrist(norm_target_pos, wrist_drawing)      
                    in_grasp_window = inside_grasp_window(dist)
                    print(f"Target distance: {dist:.2f}m | Grasp window: {in_grasp_window}")
                        
                    if in_grasp_window: # Grasping behavior: present flank to target
                        platform_cmd = present_flank()  # Orient side toward target (0° wrist yaw)

                        # Calculate progressive authority handover based on wrist yaw alignment
                        current_wrist_yaw = robot.end_of_arm.motors['wrist_yaw'].status['pos']
                        reach_auth = get_reach_authority(current_wrist_yaw, target_yaw=0.0, start_angle_deg=30, complete_angle_deg=15)  # Mix stow and visual commands for reaching behavior
                        reach_cmd = reach_for_target(dist)

                        offset = (-0.1 * reach_auth, -0.4 * reach_auth)
                        wrist_drawing = np.copy(wrist_rgb)
                        wrist_cmd = follow_target_w_wrist(norm_target_pos, wrist_drawing, offset)
                        cmd = hc.merge_override(hc.merge_mix(wrist_cmd, stow_cmd, reach_auth), cmd)
                        cmd = hc.merge_mix(reach_cmd, cmd, reach_auth)
                        yaw_error = abs(current_wrist_yaw - 0.0)  # For debug display
                        if graspable:
                            grasp_cmd = grasp_controller.get_command()
                            cmd = hc.merge_override(grasp_cmd, cmd)         # Grasping
                        print(f"Wrist yaw: {math.degrees(yaw_error):.1f}° | Authority: {reach_auth:.2f}, Graspable: {graspable}, Grasped: {grasped}")
                    else: # Approach behavior: face and move to target
                        platform_cmd = face_target()  # Start with rotation commands
                        platform_cmd.update(move_to_target(dist))  # Merge in forward motion
                    wrist_yaw_cmd = {k: v for k, v in wrist_cmd.items() if k == "wrist_yaw_counterclockwise"}
                    cmd = hc.merge_override(wrist_yaw_cmd, cmd)    # Visual yaw tracking
                else:
                    platform_cmd = scan_cmd  # No target detected - scan
            else:  # Take to drop-off
                cmd = hc.merge_override(carry_controller.get_command(), cmd)        # Carrying behavior

                if dest_dist is not None: # If destination is visible
                    head_cmd = follow_destination_w_head(norm_dest_pos, head_drawing)
                    platform_cmd = face_destination(norm_dest_pos)  # Rotate to face destination
                    platform_cmd.update(move_to_dest(dest_dist))  # Merge in forward motion
                    cmd = hc.merge_override(head_cmd, cmd)      # Destination tracking

                    in_dest_window = inside_dest_window(dest_dist)
                    if in_dest_window:
                        platform_cmd = present_flank_with_head()  # Orient side toward destination (0° head pan)
                        current_head_pan = robot.head.status['head_pan']['pos']
                        # check if head is close to -90
                        at_dropoff_position = abs(current_head_pan - math.radians(-90)) < math.radians(10)
                else:
                    print("Object grasped! Moving to stow position.")
                    platform_cmd = scan_cmd  # No destination detected - scan
        else:
            cmd = hc.merge_override(raise_controller.get_command(), cmd)        # Raise object
            raise_prog = raise_controller.get_progress(carry_controller.desired_state)["lift"]
            extend_auth = get_interpolated_authority(raise_prog, 0.8, 0.9)
            cmd = hc.merge_mix(extend_controller.get_command(), cmd, extend_auth)  # Extend arm while raising
            extend_prog = extend_controller.get_progress(raise_controller.desired_state)["arm"]
            drop_auth = get_interpolated_authority(extend_prog, 0.8, 0.9)
            cmd = hc.merge_mix(drop_controller.get_command(), cmd, drop_auth)  # Drop

        # Layer commands: lowest to highest priority
        cmd = hc.merge_override(platform_cmd, cmd)     # Platform rotation + forward motion
        cmd = hc.hybridize(cmd) # Human override
        
        controller.set_command(hc.set_limits(cmd))

        # Display
        wrist_colormap = cv2.applyColorMap(cv2.convertScaleAbs(wrist_depth, alpha=0.03), cv2.COLORMAP_JET)
        head_colormap = cv2.applyColorMap(cv2.convertScaleAbs(head_depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Wrist Cam', wrist_drawing)
        # cv2.imshow('Wrist Depth', wrist_colormap)
        cv2.imshow('Head Cam', head_drawing)
        # cv2.imshow('Head Depth', head_colormap)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
finally:
    robot.stop()
    cv2.destroyAllWindows()
    get_wrist_cam_frames.pipeline.stop()
