#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time, math
import numpy as np
import cv2
import pyrealsense2 as rs

# --- rutas a sus módulos ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_velocity_control import NormalizedVelocityControl, zero_vel
import stretch_body.robot as rb
import hybrid_control as hc

# ----------------- utilidades visión -----------------
def parse_whxfps(s: str):
    w,h,fps = s.lower().split('x'); return int(w), int(h), int(fps)

def get_color_intrinsics(active_profile):
    vsp = active_profile.get_stream(rs.stream.color).as_video_stream_profile()
    i = vsp.get_intrinsics()
    K = np.array([[i.fx, 0, i.ppx],
                  [0, i.fy, i.ppy],
                  [0, 0,   1     ]], dtype=np.float32)
    return K, (i.ppx, i.ppy), (vsp.width(), vsp.height())

# ---- rotación 90° (horaria) y conversiones asociadas ----
def rotate90_cw(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def intrinsics_rotated_90cw(W, H, fx, fy, cx, cy):
    """
    Para una rotación 90° CW:
      W' = H, H' = W
      u' = H-1 - v
      v' = u
      => fx' = fy, fy' = fx
         cx' = H-1 - cy
         cy' = cx
    """
    Wp, Hp = H, W
    fxp, fyp = fy, fx
    cxp = (H - 1) - cy
    cyp = cx
    return (fxp, fyp, cxp, cyp, Wp, Hp)

def uv_rotated_to_original_90cw(u_rot, v_rot, W_orig, H_orig):
    """
    Inversa de la rotación 90° CW:
      u' = H-1 - v,   v' = u
      => u = v',      v = H-1 - u'
    """
    u = v_rot
    v = (H_orig - 1) - u_rot
    return float(u), float(v)

def pixel_to_angles(u, v, cx, cy, fx, fy):
    yaw_err   =  math.atan((u - cx)/fx)    # + derecha
    pitch_err = -math.atan((v - cy)/fy)    # + arriba
    return yaw_err, pitch_err

def median_depth_at(depth_frame, u, v, win=2):
    if not depth_frame: return None
    W, H = depth_frame.get_width(), depth_frame.get_height()
    vals = []
    uu0, vv0 = int(round(u)), int(round(v))
    for dv in range(-win, win+1):
        for du in range(-win, win+1):
            uu = min(max(uu0+du, 0), W-1)
            vv = min(max(vv0+dv, 0), H-1)
            z = depth_frame.get_distance(uu, vv)
            if z > 0: vals.append(z)
    return float(np.median(vals)) if vals else None

def make_aruco_detector(dict_name="DICT_4X4_50"):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        def detect(img_bgr):
            corners, ids, _ = det.detectMarkers(img_bgr)
            return corners, ids
    else:
        params = aruco.DetectorParameters_create()
        def detect(img_bgr):
            corners, ids, _ = aruco.detectMarkers(img_bgr, dictionary, parameters=params)
            return corners, ids
    return detect

# ----------------- controlador de seguimiento -----------------
def follow_aruco_controller(u, v, cx, cy, fx, fy, kp=0.8, deadzone_deg=1.0, vmax=0.4):
    yaw_err, pitch_err = pixel_to_angles(u, v, cx, cy, fx, fy)  # rad
    dz = math.radians(deadzone_deg)
    if abs(yaw_err)   < dz: yaw_err   = 0.0
    if abs(pitch_err) < dz: pitch_err = 0.0
    cmd_pan  = -kp * yaw_err
    cmd_tilt =  kp * pitch_err
    cmd_pan  = max(-vmax, min(vmax, cmd_pan))
    cmd_tilt = max(-vmax, min(vmax, cmd_tilt))
    return {'head_pan_counterclockwise': cmd_pan, 'head_tilt_up': cmd_tilt}

# ----------------- principal -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Seguir ArUco 4x4 moviendo la cabeza del Stretch, con imagen rotada 90° CW antes de procesar.")
    ap.add_argument("--serial", type=str, default=None, help="Serial D435(i) (opcional)")
    ap.add_argument("--profile", type=str, default="640x360x60", help="WxHxFPS para color y depth")
    ap.add_argument("--dict", type=str, default="DICT_4X4_50", help="Diccionario ArUco (p.ej., DICT_4X4_50)")
    ap.add_argument("--id", type=int, default=18, help="Si se especifica, seguir sólo ese ID")
    ap.add_argument("--kp", type=float, default=0.8, help="Ganancia proporcional")
    ap.add_argument("--deadzone-deg", type=float, default=1.0, help="Zona muerta (grados)")
    ap.add_argument("--vmax", type=float, default=0.4, help="Velocidad máx. normalizada por eje [0..1]")
    ap.add_argument("--rate", type=float, default=20.0, help="Hz del lazo")
    ap.add_argument("--show", action="store_true", help="Mostrar ventana con overlay (imagen rotada)")
    args = ap.parse_args()

    # --- RealSense: color+depth y alinear depth->color ---
    W, H, FPS = parse_whxfps(args.profile)
    pipe, cfg = rs.pipeline(), rs.config()
    if args.serial: cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    for _ in range(10): pipe.wait_for_frames()

    # intrínsecos (originales) y tamaño original
    K, (cx, cy), (W0, H0) = get_color_intrinsics(pipe.get_active_profile())
    fx, fy = float(K[0,0]), float(K[1,1])

    # intrínsecos "virtuales" tras rotar 90° CW
    fxr, fyr, cxr, cyr, Wr, Hr = intrinsics_rotated_90cw(W0, H0, fx, fy, cx, cy)

    # detector ArUco (opera sobre la imagen rotada)
    detect = make_aruco_detector(args.dict)

    # Stretch: robot + controlador
    robot = rb.Robot(); robot.startup()
    controller = NormalizedVelocityControl(robot)

    if args.show:
        cv2.namedWindow("Aruco Servo (Rot 90° CW)", cv2.WINDOW_AUTOSIZE)

    dt = 1.0 / max(1e-3, args.rate)
    try:
        print("Servo ON (imagen rotada 90° CW). 'q' para salir.")
        while True:
            t0 = time.time()
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                time.sleep(0.001); continue

            img = np.asanyarray(color.get_data())    # (H0, W0, 3)
            img_rot = rotate90_cw(img)               # (Hr= W0, Wr= H0, 3)

            # Detección sobre la imagen rotada
            corners, ids = detect(img_rot)

            cmd = zero_vel.copy()
            info = "No marker"
            if ids is not None and len(ids) > 0:
                ids = np.asarray(ids).reshape(-1)
                # elegir índice
                sel = 0
                if args.id is not None:
                    m = np.where(ids == args.id)[0]
                    sel = int(m[0]) if len(m)>0 else None
                if sel is not None:
                    corn_r = corners[sel].reshape(4,2)     # en marco rotado
                    u_r, v_r = float(corn_r[:,0].mean()), float(corn_r[:,1].mean())

                    # convertir (u_r, v_r) rotados -> (u0, v0) originales para query de profundidad
                    u0, v0 = uv_rotated_to_original_90cw(u_r, v_r, W0, H0)

                    # profundidad (en marco original)
                    z_m = median_depth_at(depth, u0, v0, win=2)

                    # control usando intrínsecos rotados (fxr,fyr,cxr,cyr)
                    c_fb = follow_aruco_controller(u_r, v_r, cxr, cyr, fxr, fyr,
                                                   kp=args.kp, deadzone_deg=args.deadzone_deg, vmax=args.vmax)
                    cmd.update(c_fb)
                    info = f"ID={int(ids[sel])}  u'v'=({u_r:.1f},{v_r:.1f})  Z≈{(z_m if z_m else float('nan')):.3f} m"

                    if args.show:
                        cv2.polylines(img_rot, [corn_r.astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
                        cv2.circle(img_rot, (int(round(u_r)), int(round(v_r))), 3, (0,0,255), -1, cv2.LINE_AA)
                        cv2.circle(img_rot, (int(round(cxr)), int(round(cyr))), 3, (255,255,0), -1, cv2.LINE_AA)

            # enviar comando de cabeza
            controller.set_command(hc.hybridize(cmd))

            if args.show:
                cv2.putText(img_rot, info, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img_rot, info, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow("Aruco Servo (Rot 90° CW)", img_rot)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break

            rem = dt - (time.time()-t0)
            if rem > 0: time.sleep(rem)

    finally:
        if args.show: cv2.destroyAllWindows()
        controller.stop()
        robot.stop()
        pipe.stop()

if __name__ == "__main__":
    main()
