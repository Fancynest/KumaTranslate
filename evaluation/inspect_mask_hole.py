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
mask_crop = b0['bubble_mask'][by:by+bh, bx:bx+bw]

cv2.imwrite("evaluation/b0_mask_crop.png", mask_crop)

# Let's inspect rows 35 to 45
print("b0_mask_crop at y=38..42, x=50..65:")
for y in range(38, 43):
    print(f"y={y}:", mask_crop[y, 50:65])
