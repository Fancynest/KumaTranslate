import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

# Let's inspect the contour and why bubble_comp has zeros
# We know from test_extract_comp.py that bubble_comp has 1 contour
# Let's see its points or bounding rect
from evaluation.test_extract_comp import bubble_comp, contours
cnt = contours[0]
bx, by, bw, bh = cv2.boundingRect(cnt)
print(f"Contour 0 bbox: {bx}, {by}, {bw}, {bh}")
# Is the contour closing to the edge of the image?
print(f"Points touching edge? min_x={cnt[:,0,0].min()}, max_x={cnt[:,0,0].max()}, min_y={cnt[:,0,1].min()}, max_y={cnt[:,0,1].max()}")
# Does the contour have an internal boundary or is the zero region open to the outside?
test_img = np.zeros_like(bubble_comp)
cv2.drawContours(test_img, contours, -1, 255, -1)
print(f"Difference between test_img and 255 inside bounding box: zeros={np.sum(test_img[by:by+bh, bx:bx+bw] == 0)}")
