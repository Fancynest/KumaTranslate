import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector

img_path = 'temp/9dff84e4/extracted/page_001.png'
detector = BubbleDetector()
bubbles = detector.detect(img_path)

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Image shape: {h}x{w}")

for i in [0, 1]:
    b = bubbles[i]
    print(f"\n--- Bubble {i} ---")
    print(f"bbox:        {b['bbox']}")
    print(f"safe_bbox:   {b.get('safe_bbox')}")
    print(f"text_bbox:   {b.get('text_bbox')}")
    print(f"bubble_bbox: {b.get('bubble_bbox')}")
    mask = b.get('bubble_mask')
    if mask is not None:
        white_pixels = np.sum(mask > 0)
        print(f"mask shape: {mask.shape}, white pixels: {white_pixels}")
        # Crop mask to bbox
        bx, by, bw, bh = b.get('safe_bbox', b['bbox'])
        sub_mask = mask[by:by+bh, bx:bx+bw]
        print(f"mask in safe_bbox: {np.mean(sub_mask > 0)*100:.1f}% white")
