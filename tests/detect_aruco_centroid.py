#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time, math
from typing import Tuple, Optional
import numpy as np
import cv2
import pyrealsense2 as rs
import stretch_body.robot as rb

# ----------------- Utilidades -----------------
def parse_whxfps(s: str) -> Tuple[int,int,int]:
    w,h,fps = s.lower().split('x')
    return int(w), int(h), int(fps)

def depth_median_patch(depth_frame: rs.depth_frame, u: int, v: int, win: int=2) -> Optional[float]:
    """
    Devuelve la mediana de Z (metros) en una ventana (2*win+1)^2
    alrededor del píxel (u,v). Filtra Z=0 (inválidos).
    """
    if not depth_frame: return None
    W, H = depth_frame.get_width(), depth_frame.get_height()
    u0, v0 = int(u), int(v)
    vals = []
    for dv in range(-win, win+1):
        for du in range(-win, win+1):
            uu = min(max(u0+du, 0), W-1)
            vv = min(max(v0+dv, 0), H-1)
            z = depth_frame.get_distance(uu, vv)  # metros
            if z > 0:
                vals.append(z)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float32)))

def build_K_from_intrinsics(vsp: rs.video_stream_profile) -> np.ndarray:
    i = vsp.get_intrinsics()
    return np.array([[i.fx, 0.0,  i.ppx],
                     [0.0,  i.fy, i.ppy],
                     [0.0,  0.0,  1.0]], dtype=np.float32)

def get_color_intrinsics(active_profile: rs.pipeline_profile) -> Tuple[np.ndarray, Tuple[int,int], Tuple[float,float]]:
    vsp = active_profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vsp.get_intrinsics()
    K = build_K_from_intrinsics(vsp)
    size = (intr.width, intr.height)
    cx, cy = intr.ppx, intr.ppy
    return K, size, (cx, cy)

def pixel_to_angles(u: float, v: float, cx: float, cy: float, fx: float, fy: float) -> Tuple[float,float]:
    """
    Convierte un offset de píxel a un pequeño ángulo (rad) usando el modelo pinhole:
    yaw ≈ atan((u-cx)/fx), pitch ≈ -atan((v-cy)/fy).
    """
    yaw_err   = math.atan((u - cx)/fx)     # + izq / - der (convección típica)
    pitch_err = -math.atan((v - cy)/fy)    # + arriba / - abajo (imagen tiene y hacia abajo)
    return yaw_err, pitch_err

def get_aruco_detector(dict_name: str):
    # Soporta OpenCV 4.7+ (ArucoDetector) y fallback a detectMarkers antiguo
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        parameters = aruco.DetectorParameters()
        detector   = aruco.ArucoDetector(dictionary, parameters)
        def detect(img_bgr):
            corners, ids, _ = detector.detectMarkers(img_bgr)
            return corners, ids
        return detect, dictionary
    else:
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        parameters = aruco.DetectorParameters_create()
        def detect(img_bgr):
            return aruco.detectMarkers(img_bgr, dictionary, parameters=parameters)[:2]
        return detect, dictionary

# ----------------- Programa principal -----------------
def main():
    ap = argparse.ArgumentParser(description="RealSense + ArUco + Stretch head pan/tilt servoing.")
    ap.add_argument("--serial", type=str, default=None, help="Serial de la D435(i) (opcional).")
    ap.add_argument("--profile", type=str, default="848x480x30", help="Perfil WxHxFPS para color y depth.")
    ap.add_argument("--dict", type=str, default="DICT_4X4_50", help="Diccionario ArUco (p.ej., DICT_4X4_50).")
    ap.add_argument("--id", type=int, default=None, help="Si se indica, sólo sigue a ese ID.")
    ap.add_argument("--marker-length-m", type=float, default=None, help="(Opcional) lado del ArUco en metros (para mostrar distancia por PnP).")
    ap.add_argument("--rate", type=float, default=20.0, help="Frecuencia del lazo de control (Hz).")
    ap.add_argument("--kp", type=float, default=0.8, help="Ganancia proporcional para yaw/pitch (rad_cmd = kp * ang_err).")
    ap.add_argument("--step-max", type=float, default=0.05, help="Paso máximo por iteración en rad (seguridad).")
    ap.add_argument("--show", action="store_true", help="Muestra ventana OpenCV con overlay.")
    args = ap.parse_args()

    W, H, FPS = parse_whxfps(args.profile)
    # --- RealSense: color y depth con mismas dimensiones + depth->color align ---
    pipe, cfg = rs.pipeline(), rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipe.start(cfg)
    align   = rs.align(rs.stream.color)  # depth alineado a color (coordenadas comunes)

    # Warm-up
    for _ in range(10): pipe.wait_for_frames()

    # Intrínsecos de color (para convertir error de píxel a ángulo)
    active = pipe.get_active_profile()
    K, (W_act, H_act), (cx, cy) = get_color_intrinsics(active)
    fx, fy = float(K[0,0]), float(K[1,1])

    # ArUco
    detect, _dictionary = get_aruco_detector(args.dict)

    # Stretch
    robot = rb.Robot()
    robot.startup()
    # (Las articulaciones de cabeza son Dynamixel; ejecutan move_by inmediatamente, sin push_command)
    # https://dev-docs.hello-robot.com/stretch_body_api/
    # Opcional: velocidades suaves en cabeza
    robot.head.get_joint('head_pan').set_motion_params(1.0)
    robot.head.get_joint('head_tilt').set_motion_params(1.0)

    if args.show:
        cv2.namedWindow("Aruco Servo", cv2.WINDOW_AUTOSIZE)

    dt = 1.0 / max(1e-3, args.rate)
    try:
        print("Seguimiento iniciado. Presione 'q' en la ventana para salir (si --show).")
        while True:
            t0 = time.time()
            frames = pipe.wait_for_frames()
            frames = align.process(frames)

            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                time.sleep(0.001)
                continue

            color = np.asanyarray(c.get_data())
            corners, ids = detect(color)
            info_txt = "No marker"

            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                # Seleccionar el marcador con ID deseado o el primero
                idx = 0
                if args.id is not None:
                    matches = np.where(ids == args.id)[0]
                    if len(matches) > 0:
                        idx = int(matches[0])
                    else:
                        idx = None
                if idx is not None:
                    corn = corners[idx].reshape(4,2)
                    # Centróide (u,v)
                    u, v = float(np.mean(corn[:,0])), float(np.mean(corn[:,1]))
                    # Distancia Z por depth (mediana 5x5)
                    z_m = depth_median_patch(d, int(round(u)), int(round(v)), win=2)
                    # Errores angulares a partir de intrínsecos de color
                    yaw_err, pitch_err = pixel_to_angles(u, v, cx, cy, fx, fy)
                    # Comando incremental (rad) limitado
                    d_pan  = max(-args.step_max, min(args.step_max, args.kp * yaw_err))
                    d_tilt = max(-args.step_max, min(args.step_max, args.kp * pitch_err))
                    # Enviar comandos (ejecución inmediata en Dynamixel head)
                    # https://dev-docs.hello-robot.com/stretch_body_api/
                    if abs(d_pan) > 1e-4:
                        robot.head.move_by('head_pan', d_pan)
                    if abs(d_tilt) > 1e-4:
                        robot.head.move_by('head_tilt', d_tilt)

                    # Texto
                    info_txt = f"ID={int(ids[idx])}  uv=({u:.1f},{v:.1f})  yaw_err={yaw_err*180/math.pi:.2f}deg  pitch_err={pitch_err*180/math.pi:.2f}deg"
                    if z_m is not None:
                        info_txt += f"  Z≈{z_m:.3f} m"

                    # Overlay
                    if args.show:
                        cv2.polylines(color, [corn.astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
                        cv2.circle(color, (int(round(u)), int(round(v))), 3, (0,0,255), -1, cv2.LINE_AA)
                        cv2.circle(color, (int(round(cx)), int(round(cy))), 3, (255,255,0), -1, cv2.LINE_AA)

            if args.show:
                cv2.putText(color, info_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(color, info_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow("Aruco Servo", color)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            # Regular frecuencia
            t_elapsed = time.time() - t0
            if t_elapsed < dt:
                time.sleep(dt - t_elapsed)

    finally:
        if args.show:
            cv2.destroyAllWindows()
        robot.stop()
        pipe.stop()

if __name__ == "__main__":
    main()
