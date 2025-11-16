#!/usr/bin/env python3
from ultralytics import YOLO
import sys

model_path = "best.pt" if len(sys.argv) < 2 else sys.argv[1]

print(f"Loading model: {model_path}")
model = YOLO(model_path)

print("\n" + "="*50)
print("MODEL INFORMATION")
print("="*50)

print(f"\nModel type: {type(model.model).__name__}")
print(f"Task: {model.task}")

if hasattr(model, 'names'):
    print(f"\nClass names ({len(model.names)} classes):")
    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")
else:
    print("\nNo class names found")

if hasattr(model.model, 'yaml'):
    print(f"\nModel YAML keys: {list(model.model.yaml.keys())}")

print("\n" + "="*50)
