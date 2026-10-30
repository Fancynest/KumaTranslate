import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector
from src.inpainter import MangaInpainter

img = Image.open("temp/59c91ad5/extracted/page_001.png")
detector = BubbleDetector()
bubbles = detector.detect(img)

b0 = bubbles[0]
bx, by, bw, bh = b0['bbox']
mask = b0['bubble_mask']

print(f"Bubble 0 bbox: {bx}, {by}, {bw}, {bh}")
print(f"Mask shape: {mask.shape if mask is not None else None}")

# Check mask in the region of the bubble
img_np = np.array(img.convert("RGB"))
crop_orig = img_np[by:by+bh, bx:bx+bw]
cv2.imwrite("evaluation/b0_crop_orig.png", cv2.cvtColor(crop_orig, cv2.COLOR_RGB2BGR))

inpainter = MangaInpainter()
cleaned = inpainter.clean_bubbles(
    img,
    [b['bbox'] for b in bubbles],
    masks=[b.get('bubble_mask') for b in bubbles]
)
cleaned_np = np.array(cleaned)
crop_clean = cleaned_np[by:by+bh, bx:bx+bw]
cv2.imwrite("evaluation/b0_crop_clean.png", cv2.cvtColor(crop_clean, cv2.COLOR_RGB2BGR))

# Check remaining dark pixels in crop_clean
gray_clean = cv2.cvtColor(crop_clean, cv2.COLOR_RGB2GRAY)
dark_remnants = gray_clean < 200
print(f"Remaining pixels < 200 in bubble 0: {np.sum(dark_remnants)}")
y_d, x_d = np.where(dark_remnants)
if len(y_d) > 0:
    print(f"Dark pixels bounds in crop: y in [{y_d.min()}, {y_d.max()}], x in [{x_d.min()}, {x_d.max()}]")
    print(f"Sample gray values of remnants: {gray_clean[dark_remnants][:20]}")
