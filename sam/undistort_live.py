from get_cam_feeds import get_wide_cam_frames, get_head_cam_frames, get_wrist_cam_frames, stop_all_cameras
import cv2
import numpy as np

# Configuration
GET_WIDE_FRAME = lambda: get_wide_cam_frames()
GET_HEAD_FRAME = lambda: get_head_cam_frames()[0]
GET_WRIST_FRAME = lambda: get_wrist_cam_frames()[0]
GET_CAMERA_FRAME = GET_WIDE_FRAME

# Dummy camera calibration (paste real values here from calibration.py)
K = np.array([[521.2250693889947, 0.0, 310.9040486676812], [0.0, 520.4850724559025, 586.2287785837725], [0.0, 0.0, 1.0]])

# Fisheye distortion coefficients (4,)
D = np.array([-0.014501409771769221, -0.011116281429951417, 0.007577250171967382, -0.002003306340373116])

# Get image size from first frame to compute undistortion maps
print("Initializing undistortion maps...")
sample_frame = GET_CAMERA_FRAME()
h, w = sample_frame.shape[:2]

# Compute new camera matrix for undistortion
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0.0
)

# Precompute undistortion maps for faster processing
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)

print("Starting live undistortion feed...")
print("Press ESC to exit")

try:
    while True:
        # Get camera feed
        rgb_frame = GET_CAMERA_FRAME()
        
        # Undistort using precomputed maps
        undistorted = cv2.remap(rgb_frame, map1, map2, cv2.INTER_LINEAR)
        
        # Display both feeds
        cv2.imshow('Original', rgb_frame)
        cv2.imshow('Undistorted', undistorted)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
            
finally:
    cv2.destroyAllWindows()
    stop_all_cameras()
