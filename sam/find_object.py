from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
import cv2
import time
from target_finder import ArucoTargetFinder

print("Starting camera feeds...")
print("Press ESC to exit")

last_time = time.time()

GET_WIDE_FRAME = lambda: get_wide_cam_frames()
GET_HEAD_FRAME = lambda: get_head_cam_frames()[0]
GET_WRIST_FRAME = lambda: get_wrist_cam_frames()[0]
GET_FRAME = GET_HEAD_FRAME

finder = ArucoTargetFinder(
    target_ids=202,
    aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000), 
    persistence_frames=10
)

try:
    while True:
        # Get all camera feeds
        rgb_frame = GET_FRAME()
        drawing_frame = rgb_frame.copy()
        
        # Find target
        norm_pos = finder.get_normalized_target_position(rgb_frame, drawing_frame)
        print(f"Normalized Position: {norm_pos}")

        # Display feeds
        cv2.imshow('Color Camera', drawing_frame)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
