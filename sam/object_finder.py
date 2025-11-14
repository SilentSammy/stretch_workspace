@dataclass
class ImageObject:
    bbox : tuple = None  # (x_min, y_min, x_max, y_max)
    norm_bbox : tuple = None  # normalized (x_min, y_min, x_max, y_max)
    centroid : tuple = None  # (x, y)
    norm_centroid : tuple = None  # normalized (x, y)
    source : 'ObjectFinder' = None

class ObjectFinder:
    object_class = ImageObject
    
    def _find(self, rgb_image):
        pass

    def find(self, rgb_image):
        detections = self._find(rgb_image)
        if detections is None:
            return []
        
        height, width = rgb_image.shape[:2]
        results = []
        
        for detection in detections:
            if isinstance(detection, tuple):
                bbox = detection
                obj = self.object_class(bbox=bbox)
            else:
                obj = detection
                bbox = obj.bbox
            
            x_min, y_min, x_max, y_max = bbox
            
            centroid = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            
            norm_x_min = (x_min - width / 2) / (width / 2)
            norm_y_min = (y_min - height / 2) / (height / 2)
            norm_x_max = (x_max - width / 2) / (width / 2)
            norm_y_max = (y_max - height / 2) / (height / 2)
            norm_bbox = (norm_x_min, norm_y_min, norm_x_max, norm_y_max)
            
            norm_centroid_x = (centroid[0] - width / 2) / (width / 2)
            norm_centroid_y = (centroid[1] - height / 2) / (height / 2)
            norm_centroid = (norm_centroid_x, norm_centroid_y)
            
            obj.bbox = bbox
            obj.norm_bbox = norm_bbox
            obj.centroid = centroid
            obj.norm_centroid = norm_centroid
            obj.source = self
            
            results.append(obj)
        
        return results

