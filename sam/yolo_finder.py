# yolo_finder.py
import os
import cv2
import math
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from object_finder import ObjectFinder, ImageObject


def resolve_target_cls(model, desired_name: str|None, desired_id: int|None):
    names = model.names  # dict {id: name}
    if isinstance(names, dict) and len(names) == 1:
        return list(names.keys())[0]
    if desired_id is not None:
        return int(desired_id)
    if desired_name:
        for k, v in names.items():
            if str(v).lower() == desired_name.lower():
                return int(k)
    return None

@dataclass
class YoloObject(ImageObject):
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None

class YoloFinder(ObjectFinder):
    object_class = YoloObject
    
    def __init__(self,
                 weights: str = "yolov8n.pt",
                 conf: float = 0.25,
                 iou: float = 0.50,
                 imgsz: int = 640,
                 device: str | None = None,
                 class_name: str | None = None,
                 class_id: int | None = None):
        self.model = YOLO(weights)
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass

        self.target_cls = resolve_target_cls(self.model, class_name, class_id)
        self.class_name = class_name

    def _find(self, rgb_image, drawing_image=None):
        if rgb_image is None or not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3:
            return []

        # YOLO inference
        results = self.model.predict(
            source=rgb_image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=None if self.target_cls is None else [self.target_cls],
            stream=False,
            device=self.device,
            verbose=False
        )
        r0 = results[0]
        
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        
        detections = []
        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()
        cls  = r0.boxes.cls.cpu().numpy()
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(conf[i])
            cls_id = int(cls[i])
            
            # Convert bbox to polygon
            polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
            
            cname = self.model.names.get(cls_id, self.class_name)
            
            obj = YoloObject(
                polygon=polygon,
                class_id=cls_id,
                class_name=cname,
                confidence=score
            )
            detections.append(obj)
            
            # Draw on drawing_image if provided
            if drawing_image is not None:
                x1_draw, y1_draw = int(x1), int(y1)
                x2_draw, y2_draw = int(x2), int(y2)
                cv2.rectangle(drawing_image, (x1_draw, y1_draw), (x2_draw, y2_draw), (0,255,0), 2)
                label = f"{cname} {score:.2f}"
                cv2.putText(drawing_image, label, (x1_draw, y1_draw-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return detections

