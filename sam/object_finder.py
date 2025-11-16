import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class ImageObject:
    polygon : tuple = None  # ((x1,y1), (x2,y2), (x3,y3), (x4,y4), ...) vertices of bounding polygon
    norm_polygon : tuple = None  # normalized vertices
    centroid : tuple = None  # (x, y)
    norm_centroid : tuple = None  # normalized (x, y)
    source_finder : 'ObjectFinder' = None
    source_image = None  # The original image this object was detected in

class ObjectFinder:
    object_class = ImageObject
    
    def _find(self, rgb_image, drawing_image=None):
        return []  # To be implemented by subclasses

    def find(self, rgb_image, drawing_image=None):
        detections = self._find(rgb_image, drawing_image)
        if detections is None or rgb_image is None:
            return []
        
        height, width = rgb_image.shape[:2]
        results = []
        
        for detection in detections:
            if isinstance(detection, tuple) and len(detection) > 0 and isinstance(detection[0], tuple):
                polygon = detection
                obj = self.object_class(polygon=polygon)
            else:
                obj = detection
                polygon = obj.polygon
            
            # Compute centroid as average of all vertices
            centroid_x = sum(pt[0] for pt in polygon) / len(polygon)
            centroid_y = sum(pt[1] for pt in polygon) / len(polygon)
            centroid = (centroid_x, centroid_y)
            
            # Normalize polygon vertices
            norm_polygon = tuple(
                ((x - width / 2) / (width / 2), (y - height / 2) / (height / 2))
                for x, y in polygon
            )
            
            # Normalize centroid
            norm_centroid_x = (centroid[0] - width / 2) / (width / 2)
            norm_centroid_y = (centroid[1] - height / 2) / (height / 2)
            norm_centroid = (norm_centroid_x, norm_centroid_y)
            
            obj.polygon = polygon
            obj.norm_polygon = norm_polygon
            obj.centroid = centroid
            obj.norm_centroid = norm_centroid
            obj.source_finder = self
            obj.source_image = rgb_image
            
            results.append(obj)
        
        return results

@dataclass
class ArucoObject(ImageObject):
    id: int | None = None  # ArUco marker ID
    dictionary: int | None = None  # ArUco dictionary enum value

class ArucoFinder(ObjectFinder):
    object_class = ArucoObject
    
    def __init__(self, dictionary=None, ids=None):
        """
        Args:
            dictionary: ArUco dictionary enum (e.g., cv2.aruco.DICT_4X4_50)
            ids: List of marker IDs to detect, or None to detect all
        """
        if dictionary is None:
            dictionary = cv2.aruco.DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.ids_to_detect = ids
        self.dictionary_enum = dictionary
    
    def _find(self, rgb_image, drawing_image=None):
        corners, ids, _ = self.detector.detectMarkers(rgb_image)
        
        if ids is None:
            return []
        
        # Draw detected markers on the drawing image if provided
        if drawing_image is not None:
            import cv2
            cv2.aruco.drawDetectedMarkers(drawing_image, corners, ids)
        
        detections = []
        for i, marker_id in enumerate(ids.flatten()):
            # Filter by IDs if specified
            if self.ids_to_detect is not None and marker_id not in self.ids_to_detect:
                continue
            
            # Get polygon from corners (ArUco returns 4 corners)
            corner = corners[i][0]
            polygon = tuple((float(pt[0]), float(pt[1])) for pt in corner)
            
            obj = ArucoObject(polygon=polygon, id=int(marker_id), dictionary=self.dictionary_enum)
            detections.append(obj)
        
        return detections

class MultiFinder(ObjectFinder):
    def __init__(self, finders):
        self.finders = finders
    
    def _find(self, rgb_image, drawing_image=None):
        all_detections = []
        for finder in self.finders:
            detections = finder._find(rgb_image, drawing_image)
            all_detections.extend(detections)
        return all_detections

# Common Helpers
def get_largest(img_objects):
    if not img_objects:
        return None
    largest_obj = max(img_objects, key=lambda obj: cv2.contourArea(np.array(obj.polygon, dtype=np.float32)))
    return largest_obj
