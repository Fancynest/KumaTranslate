import sys
sys.path.insert(0, ".")
from PIL import Image

im = Image.open('evaluation/futaiwa_p2_translated.png')
# Crop top right: x: 350 to 590, y: 0 to 200
crop_tr = im.crop((350, 0, 590, 200))
crop_tr.save('evaluation/check_tr_p2.png')
print("Saved evaluation/check_tr_p2.png")
