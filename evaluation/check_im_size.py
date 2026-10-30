import sys
sys.path.insert(0, ".")
from PIL import Image

im = Image.open('temp/9dff84e4/extracted/page_001.png')
print("Image size:", im.size)
