import cv2
import numpy as np
import pyrealsense2 as rs

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

def get_norm_destination_pos(rgb_frame, drawing_frame=None):
    # Pseudo-static variables for aruco detector and persistence
    get_norm_destination_pos.aruco_det = aruco_det = getattr(get_norm_destination_pos, 'aruco_det', None) or cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000), cv2.aruco.DetectorParameters())
    get_norm_destination_pos.last_destination = getattr(get_norm_destination_pos, 'last_destination', None)
    get_norm_destination_pos.persistence_count = getattr(get_norm_destination_pos, 'persistence_count', 0)

    corners, ids, _ = aruco_det.detectMarkers(rgb_frame)
    
    # Try to detect destination marker (ID 12 from 4x4 dictionary)
    current_destination = None
    if ids is not None:
        markers = zip(corners, ids.flatten())
        markers = [(c, i) for c, i in markers if i == 12]  # ID 12 for destination
        if len(markers) > 0:
            marker = markers[0]
            pos = np.mean(marker[0][0], axis=0)
            
            # Draw marker visualization in different color
            if drawing_frame is not None:
                cv2.aruco.drawDetectedMarkers(drawing_frame, [marker[0]], np.array([[marker[1]]]))
                # Draw a green circle around destination marker to distinguish it
                center = (int(pos[0]), int(pos[1]))
                cv2.circle(drawing_frame, center, 30, (0, 255, 0), 3)  # Green circle
            
            # Calculate normalized position
            current_destination = ((pos[0] - rgb_frame.shape[1]/2) / (rgb_frame.shape[1]/2),
                                 (pos[1] - rgb_frame.shape[0]/2) / (rgb_frame.shape[0]/2))
    
    # Handle persistence logic
    if current_destination is not None:
        # Fresh detection - update and reset persistence
        get_norm_destination_pos.last_destination = current_destination
        get_norm_destination_pos.persistence_count = 10  # Persist for 10 calls
        return current_destination
    elif get_norm_destination_pos.persistence_count > 0:
        # Use persisted destination
        get_norm_destination_pos.persistence_count -= 1
        return get_norm_destination_pos.last_destination
    else:
        # No destination and persistence expired
        return None

try:
    print("Head camera test - Press 'Esc' to exit")
    
    while True:
        # Get frames from head camera
        rgb_image, depth_image = get_head_cam_frames()
        drawing_frame = np.copy(rgb_image)
        
        # Test destination detection
        norm_destination_pos = get_norm_destination_pos(rgb_image, drawing_frame)
        
        if norm_destination_pos is not None:
            print(f"Destination detected at: ({norm_destination_pos[0]:.3f}, {norm_destination_pos[1]:.3f})")
        
        # Create depth colormap for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Display images (use drawing_frame with markers)
        cv2.imshow('Head Cam - Color', drawing_frame)
        cv2.imshow('Head Cam - Depth', depth_colormap)
        
        # Check for exit
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

finally:
    cv2.destroyAllWindows()
    if hasattr(get_head_cam_frames, 'pipeline') and get_head_cam_frames.pipeline is not None:
        get_head_cam_frames.pipeline.stop()
        print("Head camera pipeline stopped")