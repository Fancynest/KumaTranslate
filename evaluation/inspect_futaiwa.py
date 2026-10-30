import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector
from src.inpainter import MangaInpainter

img = Image.open("temp/59c91ad5/extracted/page_001.png")
detector = BubbleDetector()
bubbles = detector.detect(img)

print(f"Total bubbles detected: {len(bubbles)}")
for i, b in enumerate(bubbles):
    x, y, w, h = b['bbox']
    mask_present = b.get('bubble_mask') is not None
    mask_pixels = np.sum(b['bubble_mask'] > 0) if mask_present else 0
    print(f"Bubble {i}: bbox=({x}, {y}, {w}, {h}), shape={b.get('shape')}, mask_pixels={mask_pixels}, safe_bbox={b.get('safe_bbox')}, text_bbox={b.get('text_bbox')}")

