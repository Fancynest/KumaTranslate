import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image

img = Image.open("temp/59c91ad5/extracted/page_001.png")
gray = np.array(img.convert("L"))
# bubble 0: bbox=(450, 0, 79, 114), text_bbox=(451, 25, 78, 68)
bx, by, bw, bh = 450, 0, 79, 114
roi_gray = gray[by:by+bh, bx:bx+bw]

bg_bright = float(np.percentile(roi_gray, 85))
thresh = max(130, int(bg_bright - 35))
bright_mask = (roi_gray >= thresh).astype(np.uint8) * 255
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bright_opened = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_opened)

print(f"num_labels in bright_opened: {num_labels}")
for lbl in range(1, num_labels):
    print(f"Label {lbl}: area={stats[lbl, cv2.CC_STAT_AREA]}, bbox={stats[lbl, :4]}")
