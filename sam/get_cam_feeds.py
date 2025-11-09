import pyrealsense2 as rs
import d405_helpers as dh
import cv2
import numpy as np

def get_head_cam_frames():
    """ Get frames from the head camera (Intel RealSense D435i) """
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
    """ Get frames from the wrist camera (Intel RealSense D405) """
    # Psuedo-static variable for realsense pipeline
    get_wrist_cam_frames.pipeline = pipeline = getattr(get_wrist_cam_frames, 'pipeline', None) or dh.start_d405(exposure='auto')[0]
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image

def get_wide_cam_frames():
    # Pseudo-static variable for wide-angle camera
    get_wide_cam_frames.cap = getattr(get_wide_cam_frames, 'cap', None)
    
    if get_wide_cam_frames.cap is None:
        # Initialize wide-angle camera (Arducam OV9782 at /dev/video6)
        cap = cv2.VideoCapture(6)
        if not cap.isOpened():
            raise RuntimeError("Failed to open wide-angle camera at /dev/video6")
        get_wide_cam_frames.cap = cap
    
    ret, frame = get_wide_cam_frames.cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from wide-angle camera")
    
    # Rotate 90 degrees counter-clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return frame

def stop_head_cam():
    """Stop and cleanup head camera pipeline"""
    if hasattr(get_head_cam_frames, 'pipeline') and get_head_cam_frames.pipeline:
        get_head_cam_frames.pipeline.stop()
        get_head_cam_frames.pipeline = None

def stop_wrist_cam():
    """Stop and cleanup wrist camera pipeline"""
    if hasattr(get_wrist_cam_frames, 'pipeline') and get_wrist_cam_frames.pipeline:
        get_wrist_cam_frames.pipeline.stop()
        get_wrist_cam_frames.pipeline = None

def stop_wide_cam():
    """Stop and cleanup wide-angle camera"""
    if hasattr(get_wide_cam_frames, 'cap') and get_wide_cam_frames.cap:
        get_wide_cam_frames.cap.release()
        get_wide_cam_frames.cap = None

def stop_all_cameras():
    """Stop and cleanup all cameras"""
    stop_head_cam()
    stop_wrist_cam()
    stop_wide_cam()

if __name__ == "__main__":
    try:
        while True:
            # Get camera feeds
            head_rgb, head_depth = get_head_cam_frames()
            wrist_rgb, wrist_depth = get_wrist_cam_frames()
            wide_rgb = get_wide_cam_frames()
            
            # Display RGB feeds
            cv2.imshow('Head Camera (D435)', head_rgb)
            cv2.imshow('Wrist Camera (D405)', wrist_rgb)
            cv2.imshow('Wide-Angle Camera (OV9782)', wide_rgb)
            
            # Exit on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
                
    finally:
        cv2.destroyAllWindows()
        stop_all_cameras()
