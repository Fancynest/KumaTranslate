import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector

img = Image.open("temp/59c91ad5/extracted/page_001.png")
gray = np.array(img.convert("L"))
detector = BubbleDetector()
bubbles = detector.detect(img)
b0 = bubbles[0]
bx, by, bw, bh = b0['bbox']
mask_full = b0['bubble_mask']

k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

eroded_mask = cv2.erode((mask_full > 0).astype(np.uint8), k_erode, iterations=1)
interior_idx = eroded_mask > 0

interior_gray = gray[interior_idx]
bg_val = int(np.percentile(interior_gray, 92))
dark_thresh = min(190, bg_val - 25)

print(f"bg_val: {bg_val}, dark_thresh: {dark_thresh}")

dark_pixels = (gray < dark_thresh) & interior_idx
print(f"dark_pixels count: {np.sum(dark_pixels)}")

dark_dilated = cv2.dilate(dark_pixels.astype(np.uint8), k_dilate, iterations=1)
dark_to_erase = (dark_dilated > 0) & interior_idx
print(f"dark_to_erase count: {np.sum(dark_to_erase)}")

# Look at the crop of dark_to_erase
crop_erase = dark_to_erase[by:by+bh, bx:bx+bw]
crop_gray = gray[by:by+bh, bx:bx+bw]
crop_mask = mask_full[by:by+bh, bx:bx+bw]
crop_eroded = eroded_mask[by:by+bh, bx:bx+bw]

# Why are there remnants? Remnants are pixels where crop_gray < 200 but not in crop_erase
remnants = (crop_gray < 200) & (~crop_erase)
print(f"Remnants count: {np.sum(remnants)}")
y_r, x_r = np.where(remnants)
print(f"Remnants sample positions (y, x):")
for i in range(min(10, len(y_r))):
    yy, xx = y_r[i], x_r[i]
    print(f"y={yy}, x={xx}, gray={crop_gray[yy, xx]}, mask={crop_mask[yy, xx]}, eroded={crop_eroded[yy, xx]}")
