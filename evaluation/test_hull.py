import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from evaluation.test_extract_comp import bubble_comp, contours

cnt = contours[0]
hull = cv2.convexHull(cnt)
test_hull = np.zeros_like(bubble_comp)
cv2.drawContours(test_hull, [hull], -1, 255, -1)

bx, by, bw, bh = cv2.boundingRect(cnt)
print(f"Zeros inside hull in bbox: {np.sum(test_hull[by:by+bh, bx:bx+bw] == 0)}")
