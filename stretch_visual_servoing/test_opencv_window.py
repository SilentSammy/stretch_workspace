#!/usr/bin/env python3
import cv2
import numpy as np

print("Testing OpenCV window display...")

# Create a simple test image
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
test_image[:] = (0, 255, 0)  # Green background
cv2.putText(test_image, 'OpenCV Test Window', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

print("Attempting to show window...")
cv2.imshow('OpenCV Test', test_image)
print("Window should be visible now")

print("Press any key to close...")
key = cv2.waitKey(0)  # Wait indefinitely for key press
print(f"Key pressed: {key}")

cv2.destroyAllWindows()
print("Test complete")