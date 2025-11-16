from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
import cv2
import time
import os
from datetime import datetime

# Configuration
GET_WIDE_FRAME = lambda: get_wide_cam_frames()
GET_HEAD_FRAME = lambda: get_head_cam_frames()[0]
GET_WRIST_FRAME = lambda: get_wrist_cam_frames()[0]
CAPTURE_INTERVAL = 3.0  # seconds between captures
GET_CAMERA_FRAME = GET_WIDE_FRAME

# Create directory for this run
run_dir = f"captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)
print(f"Saving images to: {run_dir}/")

last_save_time = time.time()
image_count = 0

try:
    while True:
        # Get camera feed
        rgb_frame = GET_CAMERA_FRAME()
        
        # Calculate countdown
        elapsed = time.time() - last_save_time
        countdown = max(0, CAPTURE_INTERVAL - elapsed)
        
        # Add countdown text to image
        display_frame = rgb_frame.copy()
        cv2.putText(display_frame, f"Next: {countdown:.1f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display feed
        cv2.imshow('Wide-Angle Camera (OV9782)', display_frame)
        
        # Save image every interval
        if countdown == 0:
            filename = f"{run_dir}/image_{image_count:04d}.jpg"
            cv2.imwrite(filename, rgb_frame)
            print(f"Saved: {filename}")
            image_count += 1
            last_save_time = time.time()
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
