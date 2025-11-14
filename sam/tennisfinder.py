# tennis_finder.py
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
from target_finder import TargetFinder 


def rotate90_cw(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def uv_rotated_to_original_90cw(u_rot: float, v_rot: float, W_orig: int, H_orig: int):
    u = v_rot
    v = (H_orig - 1) - u_rot
    return float(u), float(v)

def pick_best_detection(boxes, prefer: str = "conf"):
    if boxes is None or boxes.data is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy()
    if prefer == "area":
        areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(conf))
    return xyxy[idx], float(conf[idx]), int(cls[idx])

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
    return None  # sin filtro

class TennisFinder(TargetFinder):
    def __init__(self,
                 weights: str = "best.pt",
                 conf: float = 0.25,
                 iou: float = 0.50,
                 imgsz: int = 640,
                 device: str | None = None,
                 class_name: str | None = "tennis ball",
                 class_id: int | None = None,
                 prefer: str = "conf",                 
                 persistence_frames: int = 10,
                 smooth_alpha: float = 0.5,
                 rotate90_cw_input: bool = False):
        super().__init__(persistence_frames)
        self.model = YOLO(weights)
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass  # Ultralytics maneja fallback internamente

        self.target_cls = resolve_target_cls(self.model, class_name, class_id)
        self.class_name = class_name
        self.prefer     = prefer

        # persistencia / suavizado
        self.alpha      = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.last_centroid = None
        self.last_bbox     = None
        self.last_score    = None
        self.last_cls      = None
        self.miss_count    = 0
        self.frame_id      = 0

        # rotación opcional
        self.rotate90_cw_input = bool(rotate90_cw_input)

    def _smooth(self, new_xy: tuple[float, float] | None):
        if new_xy is None or self.last_centroid is None:
            return new_xy
        ax, ay = self.last_centroid
        bx, by = new_xy
        a = self.alpha
        return (a*bx + (1-a)*ax, a*by + (1-a)*ay)

    def _draw(self, frame: np.ndarray, bbox, centroid, label: str):
        if frame is None: 
            return
        if bbox is not None:
            x1,y1,x2,y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
        if centroid is not None:
            u,v = int(round(centroid[0])), int(round(centroid[1]))
            cv2.circle(frame, (u,v), 3, (0,0,255), -1, cv2.LINE_AA)
        if label:
            org = (10, 24)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    def _detect_target(self, rgb_frame: np.ndarray, drawing_frame: np.ndarray | None = None):
        """
        rgb_frame: BGR np.ndarray (H,W,3) en la orientación NATIVA que reciba este componente.
        Si self.rotate90_cw_input=True, se rota internamente para la inferencia y se remapean resultados a la imagen original.
        """
        self.frame_id += 1
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.ndim != 3:
            # entrada inválida: usar persistencia si aplica
            if self.miss_count < self.persistence_frames and self.last_centroid is not None:
                self.miss_count += 1
                return {
                    'found': True, 'centroid': self.last_centroid, 'bbox': self.last_bbox,
                    'score': self.last_score, 'class_id': self.last_cls,
                    'class_name': self.model.names.get(self.last_cls, self.class_name) if self.last_cls is not None else self.class_name,
                    'stale': True, 'frame_id': self.frame_id
                }
            # sin persistencia
            return {'found': False, 'centroid': None, 'bbox': None, 'score': None,
                    'class_id': None, 'class_name': None, 'stale': False, 'frame_id': self.frame_id}

        H, W = rgb_frame.shape[:2]

        # Rotación (si aplica) para inferencia
        if self.rotate90_cw_input:
            img_infer = rotate90_cw(rgb_frame)
        else:
            img_infer = rgb_frame

        # Inferencia YOLO
        results = self.model.predict(
            source=img_infer,        # np.ndarray BGR
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=None if self.target_cls is None else [self.target_cls],
            stream=False,
            device=self.device,
            verbose=False
        )
        r0 = results[0]
        best = pick_best_detection(r0.boxes, prefer=self.prefer)

        label_text = ""
        if best is not None:
            (x1,y1,x2,y2), score, cls_id = best
            # centroide en el marco usado por inferencia
            u = (x1 + x2) * 0.5
            v = (y1 + y2) * 0.5

            # Si rotamos para la inferencia, mapear a coordenadas del frame ORIGINAL
            if self.rotate90_cw_input:
                u0, v0 = uv_rotated_to_original_90cw(u, v, W_orig=W, H_orig=H)
                x1o, y1o = uv_rotated_to_original_90cw(x1, y1, W, H)
                x2o, y2o = uv_rotated_to_original_90cw(x2, y2, W, H)
                # OJO: tras el mapeo, (x1o,y1o) y (x2o,y2o) pueden quedar "cruzados"; normalízalos:
                x_min, x_max = sorted([x1o, x2o])
                y_min, y_max = sorted([y1o, y2o])
                centroid = (u0, v0)
                bbox     = (x_min, y_min, x_max, y_max)
            else:
                centroid = (u, v)
                bbox     = (x1, y1, x2, y2)

            # Suavizado (opcional)
            centroid_sm = self._smooth(centroid)

            # actualizar estado/persistencia
            self.last_centroid = centroid_sm
            self.last_bbox     = bbox
            self.last_score    = score
            self.last_cls      = cls_id
            self.miss_count    = 0

            # dibujar
            cname = self.model.names.get(cls_id, self.class_name)
            label_text = f"{cname} {score:.2f}"
            self._draw(drawing_frame if drawing_frame is not None else rgb_frame, bbox, centroid_sm, label_text)
            print(centroid_sm)
            return centroid_sm

        # --- sin detección: usar persistencia si disponible ---
        if self.miss_count < self.persistence_frames and self.last_centroid is not None:
            self.miss_count += 1
            cname = self.model.names.get(self.last_cls, self.class_name) if self.last_cls is not None else (self.class_name or "")
            label_text = f"{cname} (stale)"
            self._draw(drawing_frame if drawing_frame is not None else rgb_frame, self.last_bbox, self.last_centroid, label_text)
            print(self.last_centroid)
            return self.last_centroid

