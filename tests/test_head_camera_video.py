import argparse, time
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

def parse_whxfps(s: str):
    try:
        w, h, fps = s.lower().split('x')
        return int(w), int(h), int(fps)
    except Exception:
        raise ValueError(f"Bad profile string '{s}'. Use 'WxHxFPS', e.g. 640x360x60")

def main():
    ap = argparse.ArgumentParser(description="Record MP4 from Intel RealSense D435(i) (color/depth) for a fixed duration.")
    ap.add_argument("--serial", required=False, type=str, help="Device serial (recommended)")
    ap.add_argument("--d435-profile", default="640x360x60", help="WxHxFPS for color/depth (e.g., 848x480x30)")
    ap.add_argument("--align", action="store_true", help="Align depth to color")
    ap.add_argument("--record-color", type=str, default=None, help="Output MP4 path for color stream")
    ap.add_argument("--record-depth", type=str, default=None, help="Output MP4 path for depth (colorized)")
    ap.add_argument("--duration", type=float, required=True, help="Seconds to record (e.g., 10)")
    args = ap.parse_args()

    if not args.record_color and not args.record_depth:
        raise SystemExit("Nothing to record. Pass --record-color and/or --record-depth.")

    W, H, FPS = parse_whxfps(args.d435_profile)

    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("No RealSense devices found.")

    pipeline = rs.pipeline(ctx)
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color) if args.align else None

    for _ in range(10):
        pipeline.wait_for_frames()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_color = None
    vw_depth = None
    t_end = time.monotonic() + float(args.duration)

    try:
        while True:
            if time.monotonic() >= t_end:
                break

            frames = pipeline.wait_for_frames()
            if align:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if args.record_color and color_frame:
                color_img = np.asanyarray(color_frame.get_data())
                if vw_color is None:
                    h, w = color_img.shape[:2]
                    vw_color = cv2.VideoWriter(args.record_color, fourcc, FPS, (w, h))
                vw_color.write(color_img)

            if args.record_depth and depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())
                depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                if vw_depth is None:
                    h, w = depth_viz.shape[:2]
                    vw_depth = cv2.VideoWriter(args.record_depth, fourcc, FPS, (w, h))
                vw_depth.write(depth_viz)

    finally:
        if vw_color is not None: vw_color.release()
        if vw_depth is not None: vw_depth.release()
        pipeline.stop()

if __name__ == "__main__":
    main()
