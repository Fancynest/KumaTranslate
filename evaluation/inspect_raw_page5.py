import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector

img_path = 'temp/80f46fe3/extracted/page_001.png'
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
img_rgb = np.array(img_scaled)

detector = BubbleDetector()
raw_text_boxes = detector._detect_raw_boxes(img_rgb)
yolo_detections = detector._detect_yolo_bubbles(img_rgb)

print(f"YOLO detections ({len(yolo_detections)}):")
for i, d in enumerate(yolo_detections):
    print(f"  YOLO #{i}: {d}")

print(f"\nRaw text boxes ({len(raw_text_boxes)}):")
for i, b in enumerate(raw_text_boxes):
    print(f"  TB #{i}: {b}")
