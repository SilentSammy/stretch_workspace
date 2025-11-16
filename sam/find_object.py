from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
import cv2
import time
from object_finder import ArucoFinder, MultiFinder
from yolo_finder import YoloFinder

print("Starting camera feeds...")
print("Press ESC to exit")

last_time = time.time()

GET_WIDE_FRAME = lambda: get_wide_cam_frames()
GET_HEAD_FRAME = lambda: get_head_cam_frames()[0]
GET_WRIST_FRAME = lambda: get_wrist_cam_frames()[0]
GET_FRAME = GET_HEAD_FRAME

finder = MultiFinder([
    ArucoFinder(
        dictionary=cv2.aruco.DICT_4X4_50,
        ids=[0, 4]  # All
    ),
    YoloFinder(
        weights="tennis.pt",
        conf=0.25
    )
])

try:
    while True:
        # Get all camera feeds
        rgb_frame = GET_FRAME()
        drawing_frame = rgb_frame.copy()
        
        # Find targets
        objects = finder.find(rgb_frame, drawing_frame)
        
        if objects:
            for obj in objects:
                if hasattr(obj, 'id'):  # ArucoObject
                    print(f"ArUco ID: {obj.id}, Centroid: {obj.centroid}, Norm: {obj.norm_centroid}")
                elif hasattr(obj, 'confidence'):  # YoloObject
                    print(f"{obj.class_name}: {obj.confidence:.2f}, Centroid: {obj.centroid}, Norm: {obj.norm_centroid}")
        else:
            print("No targets found")

        # Display feeds
        cv2.imshow('Color Camera', drawing_frame)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
