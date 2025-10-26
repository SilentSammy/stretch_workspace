#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import math

fx = 322.282410
fy = 322.282410
cx = 320.818268
cy = 178.779297
W, H, FPS = 640, 360, 60

def main():
    # 1) RealSense: abrir DEPTH a 640x360
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    prof = pipe.start(cfg)

    try:
        # 2) Obtener escala de profundidad, por referencia
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()  # m/unidad
        print(f"Depth scale: {depth_scale:.8f} m/uint")

        # 3) Tomar varias muestras y usar mediana en una ventana 5x5 alrededor del centro
        u0, v0 = W // 2, H // 2
        win = 2  

        zs = []
        for _ in range(10):  # pequeño warm-up + acumulación
            frames = pipe.wait_for_frames()
            d = frames.get_depth_frame()
            if not d:
                continue
            # Extraer bloque 5x5 de distancias alrededor del centro
            vals = []
            for dv in range(-win, win + 1):
                for du in range(-win, win + 1):
                    u = min(max(u0 + du, 0), W - 1)
                    v = min(max(v0 + dv, 0), H - 1)
                    z = d.get_distance(u, v)  # metros en (u,v) según SDK
                    if z > 0:                 # descartar inválidos (0)
                        vals.append(z)
            if vals:
                zs.append(np.median(vals))

        if not zs:
            print("No se obtuvieron lecturas válidas de profundidad en el centro.")
            return

        Z = float(np.median(zs))  # metros (profundidad al centro)
        # 4) Retroproyección pinhole usando TUS intrínsecos de DEPTH
        X = (u0 - cx) / fx * Z
        Y = (v0 - cy) / fy * Z
        R = math.sqrt(X*X + Y*Y + Z*Z)  # distancia euclidiana desde el origen de cámara

        print(f"Píxel central = ({u0},{v0})")
        print(f"Profundidad Z (get_distance, m): {Z:.4f}")
        print(f"Punto 3D (m): X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}")
        print(f"Distancia euclidiana ||P|| (m): {R:.4f}")

    finally:
        pipe.stop()

if __name__ == "__main__":
    main()
