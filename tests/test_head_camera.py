import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

def parse_whxfps(s: str):
    try:
        w, h, fps = s.lower().split('x')
        return int(w), int(h), int(fps)
    except Exception:
        raise ValueError(f"Bad profile string '{s}'. Use 'WxHxFPS', e.g. 424x240x15")

def list_devices():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("No RealSense devices found.")
        return
    print("Connected RealSense devices:")
    for d in devs:
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        sn   = d.get_info(rs.camera_info.serial_number) if d.supports(rs.camera_info.serial_number) else "Unknown"
        pid  = d.get_info(rs.camera_info.product_id) if d.supports(rs.camera_info.product_id) else "Unknown"
        print(f"  - {name} | SN={sn} | PID={pid}")

def load_advanced_preset_if_any(device, preset_json_path):
    if not preset_json_path:
        return False, "No preset JSON requested."
    p = Path(preset_json_path)
    if not p.is_file():
        return False, f"Preset JSON not found: {p}"
    try:
        adv = rs.rs400_advanced_mode(device)
        if not adv.is_enabled():
            adv.toggle_advanced_mode(True)
            time.sleep(2.5)  # el dispositivo se reinicia
            return "REOPEN", "Advanced mode toggled; please reopen device."
        js = p.read_text(encoding="utf-8")
        adv.load_json(js)
        return True, f"Advanced JSON loaded: {p.name}"
    except Exception as e:
        return False, f"Advanced mode not available or failed: {e}"

def colorize_depth(depth_frame: rs.depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    return depth_colormap

def export_ply(pc: rs.pointcloud, points: rs.points, color_frame, out_path: Path):
    try:
        if color_frame is not None:
            pc.map_to(color_frame)
        points.export_to_ply(str(out_path), color_frame)
        return True, f"PLY saved to {out_path}"
    except Exception as e:
        return False, f"Failed to save PLY: {e}"

def safe_imshow(winname: str, img, gui_state: dict):
    """Intenta imshow; si falla, desactiva GUI para continuar en headless."""
    try:
        cv2.imshow(winname, img)
        return True
    except Exception as e:
        if gui_state.get("enabled", True):
            print(f"[warn] GUI backend failed ({e}). Switching to headless.")
        gui_state["enabled"] = False
        return False

def main():
    ap = argparse.ArgumentParser(description="Intel RealSense demo without ROS2")
    ap.add_argument("--list", action="store_true", help="List connected devices and exit")
    ap.add_argument("--serial", type=str, default=None, help="Target device serial")
    ap.add_argument("--device-type", choices=["d435", "d405"], default="d435",
                    help="Apply suitable stream configuration")
    ap.add_argument("--d435-profile", type=str, default="424x240x15",
                    help="WxHxFPS for D435 color/depth (default 424x240x15)")
    ap.add_argument("--d405-profile", type=str, default="480x270x15",
                    help="WxHxFPS for D405 depth (default 480x270x15)")
    ap.add_argument("--align", action="store_true", help="Align depth to color (D435 only)")
    ap.add_argument("--pointcloud", action="store_true", help="Compute point cloud")
    ap.add_argument("--save-ply", type=str, default=None, help="Path to save .ply when pressing 'p' or on auto-save")
    ap.add_argument("--preset-json", type=str, default=None, help="HighAccuracyPreset.json path")
    ap.add_argument("--imu", action="store_true", help="Enable IMU (gyro+accel) if available (D435i)")
    ap.add_argument("--output-dir", type=str, default="frames_out", help="Directory to save PNGs")
    ap.add_argument("--headless", action="store_true", help="Run without GUI windows (no imshow/waitKey).")
    ap.add_argument("--save-once", action="store_true", help="Save one set of frames and exit (headless-friendly).")
    ap.add_argument("--save-every", type=int, default=0, help="If >0, save frames every N iterations.")
    ap.add_argument("--frames", type=int, default=0, help="If >0, stop after processing this many iterations.")
    ap.add_argument("--record-color", type=str, default=None, help="Ruta MP4 para grabar el stream de color")
    ap.add_argument("--record-depth", type=str, default=None, help="Ruta MP4 para grabar el depth coloreado")
    ap.add_argument("--duration", type=float, default=0.0, help="Segundos a grabar (prioriza sobre --frames si >0)")

    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detectar GUI disponible; si no, forzar headless
    gui_state = {"enabled": not args.headless}
    if gui_state["enabled"] and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; falling back to headless.")
        gui_state["enabled"] = False

    # 1) Contexto y selección de dispositivo
    ctx = rs.context()
    dev_list = ctx.query_devices()
    if len(dev_list) == 0:
        print("No RealSense devices found.")
        sys.exit(1)

    device = None
    for d in dev_list:
        if args.serial:
            if d.supports(rs.camera_info.serial_number) and d.get_info(rs.camera_info.serial_number) == args.serial:
                device = d
                break
        else:
            device = d
            break

    if device is None:
        print(f"Device with serial={args.serial} not found.")
        sys.exit(1)

    # 2) (Opcional) cargar preset JSON en modo avanzado
    adv_status, adv_msg = load_advanced_preset_if_any(device, args.preset_json)
    print(adv_msg)
    if adv_status == "REOPEN":
        ctx = rs.context()
        time.sleep(2.0)
        dev_list = ctx.query_devices()
        if len(dev_list) == 0:
            print("Device temporarily unavailable after enabling advanced mode. Replug or wait.")
            sys.exit(1)
        device = None
        for d in dev_list:
            if args.serial:
                if d.supports(rs.camera_info.serial_number) and d.get_info(rs.camera_info.serial_number) == args.serial:
                    device = d
                    break
            else:
                device = d
                break
        if device is None:
            print("Could not re-open device after enabling advanced mode.")
            sys.exit(1)
        ok, msg = load_advanced_preset_if_any(device, args.preset_json)
        print(msg)

    # 3) Configurar pipeline
    pipeline = rs.pipeline(ctx)
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    if args.device_type == "d435":
        W, H, FPS = parse_whxfps(args.d435_profile)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        if args.imu:
            # Streams IMU (si la cámara los soporta)
            try:
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            except Exception:
                pass
            try:
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
            except Exception:
                pass

    elif args.device_type == "d405":
        W, H, FPS = parse_whxfps(args.d405_profile)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        # IR opcional para visualización
        try:
            cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
        except Exception:
            pass

    # 4) Arranque
    profile = pipeline.start(cfg)

    # 5) Alineación (solo D435 con color habilitado)
    align = rs.align(rs.stream.color) if (args.align and args.device_type == "d435") else None

    # 6) PointCloud
    pc = rs.pointcloud() if args.pointcloud else None

    if gui_state["enabled"]:
        print("Running with GUI. Focus the window to use keys: 'q' quit, 's' save PNGs, 'p' save PLY.")
    else:
        print("Running headless. Use --save-once / --save-every / --frames to control saving/exit.")

    idx = 0
    saved_once = False
    try:
        # “prime” el pipeline unas cuantas iteraciones
        for _ in range(10):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if args.device_type == "d435" else None
            ir_frame    = frames.get_infrared_frame() if args.device_type == "d405" else None

            # IMU robusto (si fue solicitado)
            if args.imu:
                gyro_frame = None
                accel_frame = None
                # Iterar frames del set y extraer motion frames
                for f in frames:
                    if f.is_motion_frame():
                        mf = f.as_motion_frame()
                        st = mf.get_profile().stream_type()
                        if st == rs.stream.gyro:
                            gyro_frame = mf
                        elif st == rs.stream.accel:
                            accel_frame = mf
                # Datos listos para usar si desea imprimir/loggear
                # if gyro_frame: g = gyro_frame.get_motion_data()
                # if accel_frame: a = accel_frame.get_motion_data()

            # Visualización / Guardado
            depth_viz = colorize_depth(depth_frame) if depth_frame else None

            if gui_state["enabled"]:
                # Mostrar ventanas
                if args.device_type == "d435" and color_frame is not None:
                    color_img = np.asanyarray(color_frame.get_data())
                    if depth_viz is not None:
                        stacked = np.hstack((color_img, depth_viz))
                        if not safe_imshow("D435: Color | Depth", stacked, gui_state):
                            # Cambió a headless, seguir abajo
                            pass
                    else:
                        if not safe_imshow("D435: Color", color_img, gui_state):
                            pass
                elif args.device_type == "d405":
                    if ir_frame is not None and depth_viz is not None:
                        ir_img = np.asanyarray(ir_frame.get_data())
                        ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
                        stacked = np.hstack((ir_rgb, depth_viz))
                        if not safe_imshow("D405: IR | Depth", stacked, gui_state):
                            pass
                    elif depth_viz is not None:
                        if not safe_imshow("D405: Depth", depth_viz, gui_state):
                            pass

                if gui_state["enabled"]:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        ts = int(time.time() * 1000)
                        if depth_viz is not None:
                            cv2.imwrite(str(output_dir / f"depth_{ts}.png"), depth_viz)
                        if args.device_type == "d435" and color_frame is not None:
                            color_img = np.asanyarray(color_frame.get_data())
                            cv2.imwrite(str(output_dir / f"color_{ts}.png"), color_img)
                        if args.device_type == "d405" and ir_frame is not None:
                            ir_img = np.asanyarray(ir_frame.get_data())
                            cv2.imwrite(str(output_dir / f"ir_{ts}.png"), ir_img)
                        print(f"Saved frames to {output_dir}")
                    elif key == ord('p') and pc is not None and args.save_ply:
                        # Calcular y exportar PLY
                        if depth_frame is not None:
                            mapped = color_frame if (args.device_type == "d435" and color_frame is not None) else None
                            if mapped is not None:
                                pc.map_to(mapped)
                            points = pc.calculate(depth_frame)
                            ok, msg = export_ply(pc, points, mapped, Path(args.save_ply))
                            print(msg)
                # si GUI falló, continuará headless abajo

            if not gui_state["enabled"]:
                # Headless: guardado automático según flags
                do_save = False
                if args.save_once and not saved_once:
                    do_save = True
                    saved_once = True
                elif args.save_every > 0 and (idx % args.save_every == 0):
                    do_save = True

                if do_save:
                    ts = int(time.time() * 1000)
                    if depth_viz is not None:
                        cv2.imwrite(str(output_dir / f"depth_{ts}.png"), depth_viz)
                    if args.device_type == "d435" and color_frame is not None:
                        color_img = np.asanyarray(color_frame.get_data())
                        cv2.imwrite(str(output_dir / f"color_{ts}.png"), color_img)
                    if args.device_type == "d405" and ir_frame is not None:
                        ir_img = np.asanyarray(ir_frame.get_data())
                        cv2.imwrite(str(output_dir / f"ir_{ts}.png"), ir_img)
                    print(f"[headless] Saved frames to {output_dir} (idx={idx})")

                    # Exportar PLY automáticamente si se especificó ruta
                    if pc is not None and args.save_ply and depth_frame is not None:
                        mapped = color_frame if (args.device_type == "d435" and color_frame is not None) else None
                        if mapped is not None:
                            pc.map_to(mapped)
                        points = pc.calculate(depth_frame)
                        ok, msg = export_ply(pc, points, mapped, Path(args.save_ply))
                        print(msg)

            idx += 1
            if args.frames > 0 and idx >= args.frames:
                break

    finally:
        if gui_state["enabled"]:
            cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main()
