import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from src.detector import BubbleDetector

img_path = 'temp/ae483818/extracted/page_0010.png'
detector = BubbleDetector()
bubbles = detector.detect(img_path)
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

print(f"Total bubbles: {len(bubbles)}")
for i, b in enumerate(bubbles):
    bx, by, bw, bh = b["bbox"]
    crop = gray[by:by+bh, bx:bx+bw]
    p90 = np.percentile(crop, 90)
    std = np.std(crop)
    # std of pixels >= p90 - 20 (background area)
    bg_pixels = crop[crop >= (p90 - 20)]
    bg_std = np.std(bg_pixels) if len(bg_pixels) > 0 else 0
    print(f"Bubble {i} ({bx},{by},{bw},{bh}): p90={p90:.1f}, full_std={std:.1f}, bg_std={bg_std:.1f}")
