import sys
sys.path.insert(0, ".")
from PIL import Image

for p in ['temp/80f46fe3/extracted/page_001.png', 'temp/8546bd80/extracted/page_001.png', 'temp/b82b4e13/extracted/page_001.png']:
    try:
        im = Image.open(p)
        print(p, im.size)
    except Exception as e:
        print(p, e)
