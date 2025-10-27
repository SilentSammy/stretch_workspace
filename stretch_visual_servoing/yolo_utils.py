"""
Utilidades para Ultralytics YOLO: detectar y devolver coordenadas de bounding box en píxeles.

Función principal:
- detect_target_from_result(result, selection='best') -> dict | None

Devuelve un diccionario:
{ 'xyxy': (x1,y1,x2,y2), 'conf': float, 'cls': int }

Suposiciones razonables:
- `result` es un objeto de Ultralytics (un elemento del generador con .boxes)
- `boxes.xyxy` ya está en coordenadas de píxeles (comportamiento de Ultralytics v8+)

"""
from typing import Optional, Dict
import numpy as np


def detect_target_from_result(r, selection: str = 'best') -> Optional[Dict]:
    """
    Extrae la bounding box objetivo desde un resultado de Ultralytics.

    Parámetros:
    - r: objeto resultado (cada iteración del generador `model.predict(..., stream=True)`)
    - selection: 'best' (mayor confianza) o 'largest' (mayor área). Opcional.

    Retorna:
    - dict con keys: 'xyxy' (tuple ints x1,y1,x2,y2), 'conf' (float), 'cls' (int)
    - None si no hay detecciones

    Ejemplo:
        out = detect_target_from_result(r, selection='best')
        if out is not None:
            x1,y1,x2,y2 = out['xyxy']

    Nota: convierte coordenadas a enteros (píxeles).
    """
    # Asegurarse de que existan cajas
    boxes = getattr(r, 'boxes', None)
    if boxes is None:
        return None

    # Obtener arrays numpy
    try:
        xyxy = boxes.xyxy.cpu().numpy()      # (N,4)
    except Exception:
        try:
            # fallback si ya es numpy
            xyxy = np.array(boxes.xyxy)
        except Exception:
            return None

    if xyxy.size == 0:
        return None

    # confidencias y clases (si están disponibles)
    conf = None
    cls = None
    try:
        conf_arr = boxes.conf.cpu().numpy()
    except Exception:
        # intentar extraer de la última col si boxes.xyxy incluye conf (no común)
        conf_arr = None

    try:
        cls_arr = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        cls_arr = None

    # Seleccionar índice según criterio
    idx = 0
    if selection == 'best' and conf_arr is not None:
        idx = int(np.argmax(conf_arr))
    elif selection == 'largest':
        areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        idx = int(np.argmax(areas))
    else:
        # default: highest confidence when possible, else first box
        if conf_arr is not None:
            idx = int(np.argmax(conf_arr))
        else:
            idx = 0

    x1, y1, x2, y2 = xyxy[idx]

    # convertir a ints (pixeles)
    x1_i, y1_i, x2_i, y2_i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    out = {
        'xyxy': (x1_i, y1_i, x2_i, y2_i),
        'conf': float(conf_arr[idx]) if conf_arr is not None else None,
        'cls': int(cls_arr[idx]) if cls_arr is not None else None
    }
    return out
