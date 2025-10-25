# Stretch Workspace - Hybrid Control System

A collaborative workspace for Hello Robot Stretch development featuring a hybrid control system that seamlessly blends algorithmic control with manual gamepad override.

## 🎮 What is Hybrid Control?

The hybrid control system allows you to:
- Run your existing algorithmic control (computer vision, navigation, etc.)
- **Instantly override** any joint with gamepad input when needed
- **Proportionally blend** algorithm commands with manual input
- Switch between full manual mode and hybrid mode on-the-fly

Perfect for research, debugging, and human-robot collaboration scenarios.

## 📁 Repository Structure

```
stretch_workspace/
├── hybrid_control/           # Core hybrid control modules
│   ├── gamepad_interface.py     # Clean gamepad abstraction
│   ├── hybrid_control.py        # Main hybrid control logic
│   └── normalized_velocity_control.py  # Robot command interface
├── stretch_visual_servoing/  # Example integration
│   ├── visual_servoing_demo_2.py  # ArUco grasping with hybrid control
│   └── ...
└── README.md                # This file
```

## 🚀 Quick Start: Adding Hybrid Control to Your Script

### Prerequisites
Your script must use the `normalized_velocity_control.py` module for robot commands.

### Step 1: Import the Hybrid Controller

```python
import sys
import os
# Add the hybrid_control directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hybrid_control'))
from hybrid_control import HybridController
```

### Step 2: Initialize the Controller

```python
# Initialize hybrid controller
hc = HybridController()

# Configure which joints can be manually controlled
# (Optional - defaults to common joints)
hc.manual_mode_joints = {
    'joint_lift': True,
    'wrist_extension': True, 
    'joint_wrist_yaw': True,
    'joint_gripper_finger_left': True,
    # Add any joints you want manual control over
}
```

### Step 3: Integrate into Your Control Loop

```python
# Your existing algorithm command
cmd = {'joint_lift': 0.1, 'wrist_extension': -0.05}  # Your algorithm's output

# Apply hybrid control (blends algorithm + gamepad input)
final_cmd = hc.hybridize(cmd)

# Send to robot (your existing code)
robot.set_velocity(final_cmd)
```

### Step 4: Handle Mode Switching (Optional)

```python
# Check if user switched to full manual mode
if hc.in_manual_mode():
    print("Manual mode active - algorithm paused")
    # Your algorithm can pause or continue running
else:
    print("Hybrid mode - algorithm + gamepad blended")
    # Your algorithm runs normally
```

## 🎮 Gamepad Controls

| Input | Function |
|-------|----------|
| **Y Button** | Toggle between Hybrid Mode ↔ Manual Mode |
| **Left Stick Y** | Lift joint (up/down) |
| **Right Stick Y** | Arm extension (in/out) |
| **Right Stick X** | Wrist yaw (left/right) |
| **D-Pad Up/Down** | Gripper open/close |
| **Left Bumper/Trigger** | Additional joint control |

**Mode Indicators:**
- 🤖 **Hybrid Mode**: Algorithm runs, gamepad can override individual joints
- 🎮 **Manual Mode**: Full gamepad control, algorithm commands ignored

## 📋 Complete Integration Example

```python
#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hybrid_control'))

from hybrid_control import HybridController
import time

def main():
    # Initialize hybrid controller
    hc = HybridController()
    
    # Your robot initialization code here
    robot = initialize_robot()  # Your existing function
    
    print("Starting hybrid control demo...")
    print("Press Y on gamepad to toggle Manual/Hybrid mode")
    
    try:
        while True:
            # Your algorithm's control logic
            algorithm_cmd = your_algorithm_function()  # Your existing function
            
            # Apply hybrid control
            final_cmd = hc.hybridize(algorithm_cmd)
            
            # Send to robot
            robot.set_velocity(final_cmd)
            
            # Optional: Check mode for algorithm behavior
            if hc.in_manual_mode():
                # Algorithm can pause, continue, or modify behavior
                pass
            
            time.sleep(0.1)  # Your control loop rate
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        hc.stop()  # Clean up gamepad resources
        robot.stop()  # Your robot cleanup

if __name__ == '__main__':
    main()
```

## 🔧 Advanced Configuration

### Custom Joint Mapping

```python
# Configure which joints are available for manual control
hc.manual_mode_joints = {
    'joint_lift': True,
    'wrist_extension': True,
    'joint_wrist_yaw': True,
    'joint_head_pan': True,      # Add head control
    'joint_head_tilt': True,
    'joint_gripper_finger_left': False,  # Disable gripper override
}
```

### Gamepad Sensitivity

```python
# Adjust control sensitivity (default values shown)
hc.gamepad_interface.sensitivity = {
    'lift': 0.3,
    'extension': 0.2,
    'yaw': 0.5,
    'gripper': 0.1,
}
```

## 🔍 Debugging Tips

1. **Test gamepad connection**: Run `hybrid_control/gamepad_interface.py` directly
2. **Check joint names**: Use `robot.status` to see available joint names
3. **Monitor commands**: Print `algorithm_cmd` vs `final_cmd` to see blending
4. **Mode confusion**: Watch for Y button presses in terminal output

## 🤝 Collaborative Development

This workspace supports multiple developers on the same robot:

1. **Clone for new team member**:
   ```bash
   cp -r stretch_workspace stretch_workspace_teammate
   cd stretch_workspace_teammate
   git config user.name "TeammateName"
   git config user.email "teammate@example.com"
   ```

2. **Work in parallel**: Each developer has their own workspace copy
3. **Merge changes**: Use standard Git workflows to merge improvements

## 📄 License

This project is part of ongoing research with Hello Robot Stretch robots.

## 🆘 Troubleshooting

**Gamepad not detected**: Ensure controller is connected and recognized by `stretch_body`
**Import errors**: Check that `hybrid_control` directory is in your Python path
**Joint name mismatches**: Verify joint names match your robot's configuration
**Mode switching not working**: Confirm Y button presses are detected in terminal output

---

**Ready to enhance your Stretch robot with hybrid control? Just follow the Quick Start guide above!** 🚀