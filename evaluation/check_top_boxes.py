import sys
sys.path.insert(0, ".")
from PIL import Image

im = Image.open('evaluation/page4_translated_test.png')
w, h = im.size
crop_top = im.crop((int(w * 0.6), 0, w, int(h * 0.35)))
crop_top.save('evaluation/page4_top_rendered.png')
print("Saved evaluation/page4_top_rendered.png")
