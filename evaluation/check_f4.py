import sys
sys.path.insert(0, ".")
from PIL import Image
from src.ocr import MultiLingualOCR

im = Image.open('temp/f4c22359/extracted/page_001.png')
print(im.size)
ocr = MultiLingualOCR()
txt = ocr.rapid_ocr(im)[0]
print("Rapid text:", [t[1] for t in txt[:5]] if txt else "None")
