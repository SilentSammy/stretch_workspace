import d405_helpers as dh
from aruco_detector import ArucoDetector
import cv2
import numpy as np

def get_aruco_pos(frame, drawing_frame = None):
    corners, ids, _ = aruco_det.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return

    # Display
    if drawing_frame is not None:
        cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
    
    # Return centroids and ids
    centroids = []
    for i, corner in enumerate(corners):
        corner = corner[0]  # Get the first (and only) element of the list
        centroid = np.mean(corner, axis=0)
        centroids.append(centroid)
    
    return np.array(centroids), ids.flatten()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

pipeline, profile = dh.start_d405(exposure='auto')
depth_scale = dh.get_depth_scale(profile)
print('depth_scale =', depth_scale)

first_frame = True
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    image = np.copy(color_image)
    if first_frame:
        depth_scale = dh.get_depth_scale(profile)
        print('depth_scale =', depth_scale)
        print()

        depth_camera_info = dh.get_camera_info(depth_frame)
        color_camera_info = dh.get_camera_info(color_frame)
        camera_info = depth_camera_info
        #camera_info = color_camera_info
        print_camera_info = True
        if print_camera_info: 
            for camera_info, name in [(depth_camera_info, 'depth'), (color_camera_info, 'color')]:
                print(name + ' camera_info:')
                print(camera_info)
                print()

        first_frame = False
    drawing_frame = np.copy(image)
    result = get_aruco_pos(image, drawing_frame)
    if result is not None:
        pos, ids = result
        pos = pos[0] if len(pos) > 0 else None
        print('Aruco pos:', pos, 'ID:', ids[0] if len(ids) > 0 else None)
    else:
        pos = None
        print('No ArUco markers detected')

    cv2.imshow('Aruco Detection', drawing_frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break