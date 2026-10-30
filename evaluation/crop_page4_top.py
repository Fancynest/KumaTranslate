import sys
sys.path.insert(0, ".")
from PIL import Image

im = Image.open('evaluation/page4_translated_test.png')
# Crop upper right
w, h = im.size
crop_top = im.crop((int(w * 0.6), 0, w, int(h * 0.3)))
crop_top.save('evaluation/page4_top_crop.png')
print("Saved evaluation/page4_top_crop.png", crop_top.size)
