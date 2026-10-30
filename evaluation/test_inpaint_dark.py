import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from PIL import Image
from src.inpainter import MangaInpainter

inpainter = MangaInpainter()
img_orig = cv2.imread('temp/ae483818/extracted/page_0014.png')
img_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))

# Bubble 0 is at (1004, 220, 253, 220)
# Let's test clean_bubbles on Bubble 0 bbox:
bboxes = [(1004, 220, 253, 220)]
cleaned_pil = inpainter.clean_bubbles(img_pil, bboxes)
cleaned_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)

# Crop bubble 0 in cleaned_cv:
c0 = cleaned_cv[220:220+220, 1004:1004+253]
cv2.imwrite('evaluation/p14_b0_cleaned.png', c0)

print("Original b0 mean:", np.mean(img_orig[220:220+220, 1004:1004+253]))
print("Cleaned b0 mean:", np.mean(c0))
