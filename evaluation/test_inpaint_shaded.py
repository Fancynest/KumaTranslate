import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from src.inpainter import MangaInpainter

img = Image.open('temp/ae483818/extracted/page_0008.png').convert("RGB")
# Bubble #8 is around x: 130..300, y: 1280..1400 (scaled coordinates: 68..150, 650..710)
# Let's crop original:
w_orig, h_orig = img.size
scale_up = 750.0 / min(w_orig, h_orig)
# Scaled bbox: (136, 1289, 154, 105)
# In scaled image (750x1059) vs original image (1475x2084):
# Notice: detector was run on scaled image!
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# Crop the shaded bubble
crop = img_scaled.crop((130, 1280, 130 + 170, 1280 + 120))
crop.save("evaluation/crop_shaded_bubble_orig.png")
print("Saved crop_shaded_bubble_orig.png")

inpainter = MangaInpainter()
b8_bbox = (136, 1289, 154, 82)
cleaned = inpainter.clean_bubbles(img_scaled, [b8_bbox])
crop_cleaned = cleaned.crop((130, 1280, 130 + 170, 1280 + 120))
crop_cleaned.save("evaluation/crop_shaded_bubble_cleaned.png")
print("Saved crop_shaded_bubble_cleaned.png")
