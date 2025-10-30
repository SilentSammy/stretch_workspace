import numpy as np

class TargetFinder:
    def __init__(self, persistence_frames=10):
        self.persistence_frames = persistence_frames
        self.last_target = None
        self.persistence_count = 0
    
    def _detect_target(self, rgb_frame, drawing_frame=None):
        pass
    
    def get_normalized_target_position(self, rgb_frame, drawing_frame=None, center_on_persistence=False):
        # Try to detect target in current frame
        pixel_target = self._detect_target(rgb_frame, drawing_frame)
        
        if pixel_target is not None:
            # Convert to normalized coordinates
            current_target = self.pixel_to_normalized(pixel_target, rgb_frame.shape)
            # Fresh detection - update and reset persistence
            self.last_target = current_target
            self.persistence_count = self.persistence_frames
            return current_target
        elif self.persistence_count > 0:
            # Use persisted target or center
            self.persistence_count -= 1
            return (0, 0) if center_on_persistence else self.last_target
        else:
            # No target and persistence expired
            return None
    
    def reset_persistence(self):
        self.last_target = None
        self.persistence_count = 0
    
    @staticmethod
    def pixel_to_normalized(pixel_pos, frame_shape):
        height, width = frame_shape[:2]
        x, y = pixel_pos
        
        norm_x = (x - width / 2) / (width / 2)
        norm_y = (y - height / 2) / (height / 2)
        
        return norm_x, norm_y
    
    @staticmethod
    def normalized_to_pixel(norm_pos, frame_shape):
        height, width = frame_shape[:2]
        norm_x, norm_y = norm_pos
        
        pixel_x = int((norm_x + 1.0) * width / 2.0)
        pixel_y = int((norm_y + 1.0) * height / 2.0)
        
        # Clamp to image bounds
        pixel_x = max(0, min(width - 1, pixel_x))
        pixel_y = max(0, min(height - 1, pixel_y))
        
        return pixel_x, pixel_y

class ArucoTargetFinder(TargetFinder):
    def __init__(self, target_ids, aruco_dict, persistence_frames=10):
        super().__init__(persistence_frames)
        self.target_ids = target_ids if isinstance(target_ids, list) else [target_ids]
        import cv2
        self.aruco_det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    def _detect_target(self, rgb_frame, drawing_frame=None):
        corners, ids, _ = self.aruco_det.detectMarkers(rgb_frame)
        
        if ids is None:
            return None
        
        # Find markers matching target IDs
        markers = zip(corners, ids.flatten())
        markers = [(c, i) for c, i in markers if i in self.target_ids]
        
        if len(markers) == 0:
            return None
        
        # Use first matching marker
        marker = markers[0]
        pos = np.mean(marker[0][0], axis=0)
        
        # Draw marker visualization
        if drawing_frame is not None:
            import cv2
            cv2.aruco.drawDetectedMarkers(drawing_frame, [marker[0]], np.array([[marker[1]]]))
        
        # Return pixel coordinates
        return tuple(pos)

