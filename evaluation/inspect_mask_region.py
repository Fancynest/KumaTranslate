import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from src.detector import BubbleDetector

img = Image.open("temp/59c91ad5/extracted/page_001.png")
detector = BubbleDetector()
bubbles = detector.detect(img)
b0 = bubbles[0]
mask = b0['bubble_mask']

print("Global coords of remnants: x in [450+53, 450+78] = [503, 528], y in [39, 78]")
# Check mask values
roi_mask = mask[39:79, 503:528]
print(f"Mask values in remnant region: min={roi_mask.min()}, max={roi_mask.max()}, sum={roi_mask.sum()}")
print(f"Unique values in remnant region: {np.unique(roi_mask)}")
print(f"Percentage of 255 in remnant region: {np.mean(roi_mask > 0) * 100:.2f}%")
for y in [40, 50, 60, 70]:
    row = mask[y, 503:528]
    print(f"y={y}: {row[:10]}")
