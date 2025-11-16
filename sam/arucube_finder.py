import cv2
import numpy as np
from object_finder import ObjectFinder, ImageObject
from dataclasses import dataclass

@dataclass
class ArucubeObject(ImageObject):
    marker_ids: list = None  # List of ArUco marker IDs that form this cube
    marker_count: int = None  # Number of visible faces

class ArucubeFinder(ObjectFinder):
    object_class = ArucubeObject
    
    def __init__(self, aruco_finder, group_by_id=True):
        """
        Args:
            aruco_finder: An ArucoFinder instance to detect markers
            group_by_id: If True, groups markers by ID. If False, treats all detected markers as one cube.
        """
        self.aruco_finder = aruco_finder
        self.group_by_id = group_by_id
    
    def _find(self, rgb_image, drawing_image=None):
        # Get all ArUco detections
        aruco_objects = self.aruco_finder._find(rgb_image, drawing_image)
        
        if not aruco_objects:
            return []
        
        # Group markers
        if self.group_by_id:
            # Group by marker ID
            groups = {}
            for obj in aruco_objects:
                marker_id = obj.id
                if marker_id not in groups:
                    groups[marker_id] = []
                groups[marker_id].append(obj)
        else:
            # Treat all markers as one cube
            groups = {'all': aruco_objects}
        
        # Create bounding polygon for each group
        cubes = []
        for group_key, markers in groups.items():
            # Collect all vertices from all markers in this group
            all_vertices = []
            marker_ids = []
            for marker in markers:
                all_vertices.extend(marker.polygon)
                marker_ids.append(marker.id)
            
            # Convert to numpy array for convex hull computation
            points = np.array(all_vertices, dtype=np.float32)
            
            # Compute convex hull to get bounding polygon
            hull = cv2.convexHull(points)
            
            # Convert hull back to tuple of tuples
            bounding_polygon = tuple((float(pt[0][0]), float(pt[0][1])) for pt in hull)
            
            # Create ArucubeObject
            cube = ArucubeObject(
                polygon=bounding_polygon,
                marker_ids=marker_ids,
                marker_count=len(markers)
            )
            cubes.append(cube)
        
        return cubes
