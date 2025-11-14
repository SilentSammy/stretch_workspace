from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
from non_blocking_poller import NonBlockingPoller
import cv2
import time

print("Starting camera feeds...")
print("Press ESC to exit")

last_time = time.time()

GET_WIDE_FRAME = get_wide_cam_frames
GET_HEAD_FRAME = get_head_cam_frames
GET_WRIST_FRAME = get_wrist_cam_frames

GET_WIDE_FRAME = NonBlockingPoller(GET_WIDE_FRAME).get
GET_HEAD_FRAME = NonBlockingPoller(GET_HEAD_FRAME).get
GET_WRIST_FRAME = NonBlockingPoller(GET_WRIST_FRAME).get

try:
    while True:
        # Get all camera feeds
        wide_frame = GET_WIDE_FRAME()
        head_rgb, head_depth = GET_HEAD_FRAME()
        wrist_rgb, wrist_depth = GET_WRIST_FRAME()
        
        # Calculate time since last iteration
        current_time = time.time()
        iteration_time = current_time - last_time
        last_time = current_time
        
        # Display iteration time
        print(f"Iteration time: {iteration_time:.4f}s ({1/iteration_time:.1f} fps)")
        
        # Display feeds
        cv2.imshow('Wide Camera', wide_frame)
        cv2.imshow('Head Camera', head_rgb)
        cv2.imshow('Wrist Camera', wrist_rgb)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
