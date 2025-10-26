import argparse, sys
import numpy as np
import cv2
import pyrealsense2 as rs

def parse_whxfps(s: str):
    try:
        w, h, fps = s.lower().split("x")
        return int(w), int(h), int(fps)
    except Exception:
        raise ValueError(f"Formato inválido '{s}'. Use 'WxHxFPS' (p. ej., 848x480x30).")

def main():
    ap = argparse.ArgumentParser(
        description="Mostrar en OpenCV el stream RGB de la RealSense (cabeza Stretch 3)."
    )
    ap.add_argument("--serial", type=str, default=239122073909, help="Número de serie (opcional).")
    ap.add_argument("--profile", type=str, default="848x480x30", help="Perfil WxHxFPS (p. ej., 640x360x60).")
    ap.add_argument("--window", type=str, default="Stretch Head Camera (RealSense RGB)", help="Nombre de la ventana.")
    args = ap.parse_args()

    W, H, FPS = parse_whxfps(args.profile)

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("No hay cámaras RealSense conectadas."); sys.exit(1)

    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)  # Stream RGB en BGR8 (OpenCV-friendly)

    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"Error al iniciar la cámara: {e}")
        sys.exit(1)

    # Pequeño warm-up para estabilizar autoexposición
    for _ in range(10):
        pipe.wait_for_frames()

    cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)
    print(f"Mostrando {W}x{H}@{FPS}. Cierre con 'q' dentro de la ventana.")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            cv2.imshow(args.window, img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        pipe.stop()

if __name__ == "__main__":
    main()
