import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector
from src.inpainter import MangaInpainter

det = BubbleDetector()
inp = MangaInpainter()

img_path = "temp/59c91ad5/extracted/page_001.png"
img = Image.open(img_path)
bubbles = det.detect(img)
print(f"Detected {len(bubbles)} bubbles.")

b1 = bubbles[0]
print(f"Bubble 1: bbox={b1['bbox']}, text_bbox={b1['text_bbox']}, safe_bbox={b1['safe_bbox']}")
mask = b1.get("bubble_mask")
if mask is not None:
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) > 0:
        print(f"Mask range: Y=[{y_idx.min()}, {y_idx.max()}], X=[{x_idx.min()}, {x_idx.max()}]")
    else:
        print("Mask is all zeros!")
else:
    print("Mask is None!")

# Test clean_bubbles on Bubble 1
cleaned = inp.clean_bubbles(img, [b1["text_bbox"]], masks=[b1.get("bubble_mask")])
cleaned.save("evaluation/test_cleaned_b1.png")
print("Saved evaluation/test_cleaned_b1.png")
