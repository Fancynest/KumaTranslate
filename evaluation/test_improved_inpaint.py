import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from evaluation.test_extract_comp import bubble_comp, contours

img = Image.open("temp/59c91ad5/extracted/page_001.png")
gray = np.array(img.convert("L"))
clean_np = np.array(img.convert("RGB"))

# Bubble 0: bbox=(450, 0, 79, 114)
bx, by, bw, bh = 450, 0, 79, 114

# Let's test using convex hull on contour of bubble_comp
cnt = contours[0]
hull = cv2.convexHull(cnt)
mask_roi = np.zeros_like(bubble_comp)
cv2.drawContours(mask_roi, [hull], -1, 255, -1)

# Now place in full mask:
h_img, w_img = gray.shape
mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
# rx1=441, ry1=0, rx2=538, ry2=130
rx1, ry1, rx2, ry2 = 441, 0, 538, 130
mask_full[ry1:ry2, rx1:rx2] = mask_roi

# Inpainter logic with improved threshold
k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

eroded_mask = cv2.erode((mask_full > 0).astype(np.uint8), k_erode, iterations=1)
interior_idx = eroded_mask > 0

interior_gray = gray[interior_idx]
bg_val = int(np.percentile(interior_gray, 92))
fill_color = [bg_val, bg_val, bg_val]

dark_thresh = min(225, bg_val - 15)
dark_pixels = (gray < dark_thresh) & interior_idx
dark_dilated = cv2.dilate(dark_pixels.astype(np.uint8), k_dilate, iterations=1)
dark_to_erase = (dark_dilated > 0) & interior_idx

clean_np[dark_to_erase] = fill_color

# Check remnants in bubble 0
crop_clean = clean_np[by:by+bh, bx:bx+bw]
crop_gray = cv2.cvtColor(crop_clean, cv2.COLOR_RGB2GRAY)
# Exclude borders: inset by 3px
interior_box = crop_gray[3:-3, 3:-3]
remnants = np.sum(interior_box < 200)
print(f"Improved inpainting: bg_val={bg_val}, dark_thresh={dark_thresh}")
print(f"Remaining dark pixels inside bubble 0 box: {remnants}")
cv2.imwrite("evaluation/improved_b0_clean.png", cv2.cvtColor(crop_clean, cv2.COLOR_RGB2BGR))
