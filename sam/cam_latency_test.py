from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
import cv2
import time

GET_WIDE_FRAME = lambda: get_wide_cam_frames()
GET_HEAD_FRAME = lambda: get_head_cam_frames()[0]
GET_WRIST_FRAME = lambda: get_wrist_cam_frames()[0]
GET_CAMERA_FRAME = GET_WRIST_FRAME

print("Starting camera feed...")
print("Press ESC to exit")

last_time = time.time()

try:
    while True:
        # Get wide camera feed
        rgb_frame = GET_CAMERA_FRAME()
        
        # Calculate time since last iteration
        current_time = time.time()
        iteration_time = current_time - last_time
        last_time = current_time
        
        # Display iteration time
        print(f"Iteration time: {iteration_time:.4f}s ({1/iteration_time:.1f} fps)")
        
        # Display feed
        cv2.imshow('Wide Camera', rgb_frame)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
