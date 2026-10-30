import sys
sys.path.insert(0, '.')
import os
import cv2
import numpy as np
from src.detector import BubbleDetector

detector = BubbleDetector()

candidates = []
folders = ['temp/a01451f8/extracted', 'temp/ae483818/extracted']

for folder in folders:
    if not os.path.exists(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.png'):
            continue
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        bubbles = detector.detect(fpath)
        for i, b in enumerate(bubbles):
            bx, by, bw, bh = b['bbox']
            crop = img[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                continue
            mean_val = float(np.mean(crop))
            med_val = float(np.median(crop))
            p20 = float(np.percentile(crop, 20))
            p80 = float(np.percentile(crop, 80))
            if med_val < 130 or mean_val < 130 or p20 < 60 and p80 > 150:
                candidates.append((fpath, i, b['bbox'], mean_val, med_val, p20, p80, b.get('source'), b.get('type')))

print(f"Found {len(candidates)} dark/contrast bubble candidates:")
for c in candidates[:15]:
    print(c)
