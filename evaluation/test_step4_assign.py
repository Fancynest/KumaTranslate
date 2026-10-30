import sys
sys.path.insert(0, ".")
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
yolo_bubbles = [d[:4] for d in yolo_detections if d[4] == "text_bubble"]

bubble_texts = [[] for _ in range(len(yolo_bubbles))]
for tb in raw_text_boxes:
    tx, ty, tw, th = tb
    text_area = tw * th
    best_containment = 0.0
    best_bubble_idx = -1
    for b_idx, yb in enumerate(yolo_bubbles):
        bx, by, bw, bh = yb
        xi1 = max(tx, bx)
        yi1 = max(ty, by)
        xi2 = min(tx + tw, bx + bw)
        yi2 = min(ty + th, by + bh)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        containment = inter / text_area if text_area > 0 else 0.0
        if containment > best_containment:
            best_containment = containment
            best_bubble_idx = b_idx
    if best_containment >= 0.60 and best_bubble_idx >= 0:
        bubble_texts[best_bubble_idx].append(tb)

print("Step 4 assignments:")
for i in [5, 12]:
    print(f"YOLO #{i} {yolo_bubbles[i]}: assigned {len(bubble_texts[i])} text boxes:")
    for tb in bubble_texts[i]:
        print(f"   {tb}")
