import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image

img = Image.open("temp/59c91ad5/extracted/page_001.png")
gray = np.array(img.convert("L"))
h_img, w_img = gray.shape

# Bubble 0:
# yb = (445, 0, 89, 124)
# text_bbox = (451, 25, 78, 68)
bx, by, bw, bh = 445, 0, 89, 124
tx, ty, tw, th = 451, 25, 78, 68

pad_x = max(4, int(bw * 0.05))
pad_y = max(4, int(bh * 0.05))
rx1 = max(0, min(bx, tx) - pad_x)
ry1 = max(0, min(by, ty) - pad_y)
rx2 = min(w_img, max(bx + bw, tx + tw) + pad_x)
ry2 = min(h_img, max(by + bh, ty + th) + pad_y)

roi_gray = gray[ry1:ry2, rx1:rx2]
rh, rw = roi_gray.shape

rel_tx = max(0, tx - rx1)
rel_ty = max(0, ty - ry1)
rel_tw = min(rw - rel_tx, tw)
rel_th = min(rh - rel_ty, th)

text_roi = roi_gray[rel_ty:rel_ty+rel_th, rel_tx:rel_tx+rel_tw]
bg_bright = float(np.percentile(text_roi, 85))
thresh = max(130, int(bg_bright - 35))
bright_mask = (roi_gray >= thresh).astype(np.uint8) * 255
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bright_opened = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_opened)

print(f"rx1={rx1}, ry1={ry1}, rx2={rx2}, ry2={ry2}, rh={rh}, rw={rw}")
print(f"bg_bright={bg_bright}, thresh={thresh}, num_labels={num_labels}")
for lbl in range(1, num_labels):
    print(f"Label {lbl}: stats={stats[lbl]}")

text_labels = labels[rel_ty:rel_ty+rel_th, rel_tx:rel_tx+rel_tw]
non_zero = text_labels[text_labels > 0]
counts = np.bincount(non_zero)
best_label = int(np.argmax(counts))
print(f"best_label: {best_label}")

bubble_comp = (labels == best_label).astype(np.uint8) * 255
bubble_comp = cv2.morphologyEx(bubble_comp, cv2.MORPH_DILATE, kernel_open)
bubble_comp = cv2.bitwise_and(bubble_comp, bright_mask)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bubble_comp = cv2.morphologyEx(bubble_comp, cv2.MORPH_CLOSE, kernel_close)
contours, _ = cv2.findContours(bubble_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours: {len(contours)}")
if contours:
    cv2.drawContours(bubble_comp, contours, -1, 255, -1)

# Now check bubble_comp at the remnant positions (global: x in 503..528, y in 39..78)
# In roi coordinates: rel_x = 503 - rx1, rel_y = 39 - ry1
rel_x_start = 503 - rx1
rel_y_start = 39 - ry1
print(f"roi coords: rel_x_start={rel_x_start}, rel_y_start={rel_y_start}")
print("bubble_comp at rel_y_start..+3, rel_x_start..+15:")
for dy in range(4):
    print(f"dy={dy}:", bubble_comp[rel_y_start+dy, rel_x_start:rel_x_start+15])
