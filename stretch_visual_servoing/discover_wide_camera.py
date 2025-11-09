#!/usr/bin/env python3
"""
Script to discover and test the wide-angle camera on Stretch's head
"""
import cv2
import subprocess

def list_video_devices():
    """List all video devices"""
    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                          capture_output=True, text=True)
    print("Available video devices:")
    print(result.stdout)
    return result.stdout

def test_camera(device_id):
    """Test opening a camera device"""
    print(f"\nTesting camera at /dev/video{device_id}...")
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"  ❌ Failed to open /dev/video{device_id}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  ✅ Camera opened successfully")
    print(f"     Resolution: {width}x{height}")
    print(f"     FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"     Frame shape: {frame.shape}")
        # Save test image
        cv2.imwrite(f'/home/hello-robot/Desktop/test_camera_{device_id}.jpg', frame)
        print(f"     Saved test image to Desktop/test_camera_{device_id}.jpg")
    else:
        print(f"  ⚠️  Could not read frame")
    
    cap.release()
    return ret

if __name__ == "__main__":
    print("=" * 60)
    print("Stretch Wide-Angle Camera Discovery Tool")
    print("=" * 60)
    
    # List all devices
    devices_info = list_video_devices()
    
    # Extract video device numbers
    import re
    video_devices = re.findall(r'/dev/video(\d+)', devices_info)
    video_devices = sorted(set(video_devices))
    
    print(f"\nFound {len(video_devices)} video device(s): {', '.join([f'/dev/video{d}' for d in video_devices])}")
    
    # Test each device
    print("\n" + "=" * 60)
    print("Testing each camera...")
    print("=" * 60)
    
    working_cameras = []
    for device_id in video_devices:
        if test_camera(int(device_id)):
            working_cameras.append(device_id)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Working cameras: {', '.join([f'/dev/video{d}' for d in working_cameras])}")
    
    if working_cameras:
        print("\nCheck the Desktop for test images to identify which camera is the wide-angle!")
