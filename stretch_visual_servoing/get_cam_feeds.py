import pyrealsense2 as rs
import d405_helpers as dh
import cv2
import numpy as np

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

try:
    while True:
        # Get camera feeds
        head_rgb, head_depth = get_head_cam_frames()
        wrist_rgb, wrist_depth = get_wrist_cam_frames()
        
        # Display RGB feeds
        cv2.imshow('Head Camera', head_rgb)
        cv2.imshow('Wrist Camera', wrist_rgb)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    if hasattr(get_head_cam_frames, 'pipeline') and get_head_cam_frames.pipeline:
        get_head_cam_frames.pipeline.stop()
    if hasattr(get_wrist_cam_frames, 'pipeline') and get_wrist_cam_frames.pipeline:
        get_wrist_cam_frames.pipeline.stop()
