import stretch_body.gamepad_controller as gc

# Analog axes - return values from -1.0 to 1.0 (triggers: 0.0 to 1.0)
ANALOG_AXES = [
    "left_stick_x",          # Left joystick horizontal (-1.0 to 1.0)
    "left_stick_y",          # Left joystick vertical (-1.0 to 1.0)
    "right_stick_x",         # Right joystick horizontal (-1.0 to 1.0)
    "right_stick_y",         # Right joystick vertical (-1.0 to 1.0)
    "left_trigger_pulled",   # Left trigger pressure (0.0 to 1.0)
    "right_trigger_pulled"   # Right trigger pressure (0.0 to 1.0)
]

# Digital buttons - return 0 or 1
DIGITAL_BUTTONS = [
    "left_button_pressed",              # X button (Xbox) / Square (PlayStation)
    "bottom_button_pressed",            # A button (Xbox) / X (PlayStation)
    "right_button_pressed",             # B button (Xbox) / Circle (PlayStation)
    "top_button_pressed",               # Y button (Xbox) / Triangle (PlayStation)
    "left_shoulder_button_pressed",     # LB/L1 shoulder button
    "right_shoulder_button_pressed",    # RB/R1 shoulder button
    "start_button_pressed",             # Start/Options button
    "select_button_pressed",            # Back/Share button
    "left_stick_button_pressed",        # Left stick click (L3)
    "right_stick_button_pressed",       # Right stick click (R3)
    "middle_led_ring_button_pressed",   # Xbox guide/PlayStation button
    "top_pad_pressed",                  # D-pad up
    "bottom_pad_pressed",               # D-pad down
    "left_pad_pressed",                 # D-pad left
    "right_pad_pressed"                 # D-pad right
]

# Global gamepad controller instance
_gamepad = None

def init_gamepad():
    global _gamepad
    if _gamepad is None:
        _gamepad = gc.GamePadController()
        _gamepad.daemon = True
        _gamepad.start()
    return _gamepad

def get_axis(axis_name):
    if _gamepad is None:
        init_gamepad()
    return float(_gamepad.gamepad_state[axis_name])

def stop_gamepad():
    global _gamepad
    if _gamepad is not None:
        _gamepad.stop()
        _gamepad = None

def get_all_axes():
    if _gamepad is None:
        init_gamepad()
    
    result = {}
    for axis in ANALOG_AXES:
        result[axis] = float(_gamepad.gamepad_state[axis])
    for button in DIGITAL_BUTTONS:
        result[button] = int(_gamepad.gamepad_state[button])
    
    return result

def is_analog(axis_name):
    return axis_name in ANALOG_AXES

def is_digital(axis_name):
    return axis_name in DIGITAL_BUTTONS

def list_axes():
    return {
        'analog': ANALOG_AXES.copy(),
        'digital': DIGITAL_BUTTONS.copy()
    }

if __name__ == "__main__":
    # Demo/test code
    print("Gamepad Interface Demo")
    print("Available axes:")
    print(f"  Analog ({len(ANALOG_AXES)}): {ANALOG_AXES}")
    print(f"  Digital ({len(DIGITAL_BUTTONS)}): {DIGITAL_BUTTONS}")
    
    try:
        init_gamepad()
        print("\nPress Ctrl+C to exit")
        print("Move gamepad controls to see values...")
        
        import time
        while True:
            # Show only non-zero values to reduce clutter
            values = get_all_axes()
            active_values = {k: f"{v:4.1f}" for k, v in values.items() if abs(v) > 0.01}
            
            if active_values:
                print(f"Active: {active_values}")
            else:
                print("No active inputs...")
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_gamepad()
        print("Done!")