#!/usr/bin/env python3
"""
Script to discover and test the wide-angle head camera on Stretch robot.
This will list all available video devices and help identify the wide-angle camera.
"""

import cv2
import subprocess
import re

def list_video_devices():
    """List all video devices using v4l2"""
    print("=" * 60)
    print("VIDEO DEVICES DETECTED")
    print("=" * 60)
    
    try:
        # Get list of video devices
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True)
        print(result.stdout)
        
        # Parse device paths
        devices = re.findall(r'/dev/video\d+', result.stdout)
        return sorted(set(devices))
    except FileNotFoundError:
        print("v4l2-ctl not found. Installing v4l-utils...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'v4l-utils'])
        return list_video_devices()

def get_device_info(device_path):
    """Get detailed information about a video device"""
    try:
        result = subprocess.run(['v4l2-ctl', '--device=' + device_path, '--all'],
                              capture_output=True, text=True, timeout=2)
        return result.stdout
    except Exception as e:
        return f"Error getting info: {e}"

def test_camera(device_path, device_num, save_images=True):
    """Try to open and capture from a camera device"""
    print(f"\n{'=' * 60}")
    print(f"Testing {device_path} (Device #{device_num})")
    print(f"{'=' * 60}")
    
    # Get device info
    info = get_device_info(device_path)
    
    # Look for camera name/model in the info
    name_match = re.search(r'Card type\s*:\s*(.+)', info)
    camera_name = "Unknown"
    if name_match:
        camera_name = name_match.group(1)
        print(f"Camera Name: {camera_name}")
    
    # Look for resolution capabilities
    formats = re.findall(r'Size: Discrete (\d+x\d+)', info)
    if formats:
        print(f"Supported Resolutions: {', '.join(formats[:5])}")
    
    # Try to open the camera
    cap = cv2.VideoCapture(device_num)
    
    if not cap.isOpened():
        print(f"❌ Could not open {device_path}")
        return False, None
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Try to capture a frame
    ret, frame = cap.read()
    if ret:
        print(f"   Frame captured: {frame.shape}")
        
        # Save the frame to file
        if save_images:
            filename = f"camera_{device_num}_{camera_name.replace(' ', '_').replace('/', '_')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   💾 Saved sample image: {filename}")
    else:
        print(f"   ⚠️  Could not capture frame")
    
    cap.release()
    return ret, camera_name

def find_wide_angle_camera():
    """Main function to find the wide-angle camera"""
    print("\n🔍 STRETCH ROBOT - WIDE ANGLE CAMERA FINDER")
    print("=" * 60)
    
    # List all video devices
    devices = list_video_devices()
    
    if not devices:
        print("❌ No video devices found!")
        return
    
    print(f"\n📹 Found {len(devices)} video device(s)")
    print("=" * 60)
    
    # Test each device
    working_cameras = []
    for device_path in devices:
        # Extract device number from path (e.g., /dev/video0 -> 0)
        device_num = int(re.search(r'\d+', device_path).group())
        
        try:
            success, camera_name = test_camera(device_path, device_num)
            if success:
                working_cameras.append((device_path, device_num, camera_name))
        except Exception as e:
            print(f"❌ Error testing {device_path}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Working cameras: {len(working_cameras)}")
    for device_path, device_num, camera_name in working_cameras:
        print(f"  • {device_path} (index {device_num}) - {camera_name}")
    
    print("\n💡 TIP: The wide-angle camera is typically:")
    print("   - Named 'Arducam OV9782' or similar")
    print("   - Higher resolution (e.g., 1280x800)")
    print("   - NOT a RealSense device")
    
    # Identify likely wide-angle camera
    for device_path, device_num, camera_name in working_cameras:
        if 'Arducam' in camera_name or 'OV9782' in camera_name:
            print(f"\n🎯 LIKELY WIDE-ANGLE CAMERA FOUND:")
            print(f"   Device: {device_path}")
            print(f"   Index: {device_num}")
            print(f"   Name: {camera_name}")
    
    return working_cameras

if __name__ == "__main__":
    try:
        cameras = find_wide_angle_camera()
        
        print("\n" + "=" * 60)
        print("To use a camera in your code:")
        print("=" * 60)
        if cameras:
            device_path, device_num, camera_name = cameras[0]
            print(f"""
import cv2

# Open camera by device number
cap = cv2.VideoCapture({device_num})

# Or open by device path
cap = cv2.VideoCapture('{device_path}')

# Capture frame
ret, frame = cap.read()
if ret:
    print(f"Captured frame: {{frame.shape}}")
    # Process or save frame
    cv2.imwrite('frame.jpg', frame)

cap.release()
""")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()
