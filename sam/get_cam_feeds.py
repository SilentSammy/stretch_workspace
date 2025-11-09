import pyrealsense2 as rs
import d405_helpers as dh
import cv2
import numpy as np
from non_blocking_poller import NonBlockingPoller

def get_head_cam_frames(non_blocking=True):
    """ Get frames from the head camera (Intel RealSense D435i) """
    get_head_cam_frames.pipeline = getattr(get_head_cam_frames, 'pipeline', None)
    get_head_cam_frames.poller = getattr(get_head_cam_frames, 'poller', None)
    
    if get_head_cam_frames.pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        pipeline.start(config)
        get_head_cam_frames.pipeline = pipeline
    
    def poll():
        frames = get_head_cam_frames.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
        return color_image, depth_image
    
    if non_blocking:
        if get_head_cam_frames.poller is None:
            get_head_cam_frames.poller = NonBlockingPoller(poll)
        return get_head_cam_frames.poller.get()
    else:
        return poll()

def get_wrist_cam_frames(non_blocking=True):
    """ Get frames from the wrist camera (Intel RealSense D405) """
    get_wrist_cam_frames.pipeline = getattr(get_wrist_cam_frames, 'pipeline', None)
    get_wrist_cam_frames.poller = getattr(get_wrist_cam_frames, 'poller', None)
    
    if get_wrist_cam_frames.pipeline is None:
        get_wrist_cam_frames.pipeline = dh.start_d405(exposure='auto')[0]
    
    def poll():
        frames = get_wrist_cam_frames.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image
    
    if non_blocking:
        if get_wrist_cam_frames.poller is None:
            get_wrist_cam_frames.poller = NonBlockingPoller(poll)
        return get_wrist_cam_frames.poller.get()
    else:
        return poll()

def get_wide_cam_frames(non_blocking=True):
    get_wide_cam_frames.cap = getattr(get_wide_cam_frames, 'cap', None)
    get_wide_cam_frames.poller = getattr(get_wide_cam_frames, 'poller', None)
    
    if get_wide_cam_frames.cap is None:
        cap = cv2.VideoCapture(6)
        if not cap.isOpened():
            raise RuntimeError("Failed to open wide-angle camera at /dev/video6")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        get_wide_cam_frames.cap = cap
    
    def poll():
        ret, frame = get_wide_cam_frames.cap.read()
        if ret:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return None
    
    if non_blocking:
        if get_wide_cam_frames.poller is None:
            get_wide_cam_frames.poller = NonBlockingPoller(poll)
        return get_wide_cam_frames.poller.get().copy()
    else:
        return poll()

def stop_head_cam():
    """Stop and cleanup head camera pipeline"""
    if hasattr(get_head_cam_frames, 'poller') and get_head_cam_frames.poller:
        get_head_cam_frames.poller.stop()
        get_head_cam_frames.poller = None
    if hasattr(get_head_cam_frames, 'pipeline') and get_head_cam_frames.pipeline:
        get_head_cam_frames.pipeline.stop()
        get_head_cam_frames.pipeline = None

def stop_wrist_cam():
    """Stop and cleanup wrist camera pipeline"""
    if hasattr(get_wrist_cam_frames, 'poller') and get_wrist_cam_frames.poller:
        get_wrist_cam_frames.poller.stop()
        get_wrist_cam_frames.poller = None
    if hasattr(get_wrist_cam_frames, 'pipeline') and get_wrist_cam_frames.pipeline:
        get_wrist_cam_frames.pipeline.stop()
        get_wrist_cam_frames.pipeline = None

def stop_wide_cam():
    """Stop and cleanup wide-angle camera"""
    if hasattr(get_wide_cam_frames, 'poller') and get_wide_cam_frames.poller:
        get_wide_cam_frames.poller.stop()
        get_wide_cam_frames.poller = None
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
