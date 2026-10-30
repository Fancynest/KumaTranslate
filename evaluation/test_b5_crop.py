import sys
sys.path.insert(0, ".")
import cv2
from PIL import Image
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

ocr = MultiLingualOCR()
img = Image.open('temp/9dff84e4/extracted/page_001.png')

# Bubble 5 bbox: [589, 615, 98, 129]
crop = img.crop((589, 615, 589 + 98, 615 + 129))
raw_rapid = ocr.read_text(crop, lang_hint="eng_Latn")
raw_sec = ocr.read_text_secondary(crop, lang="en")
print("RapidOCR:", repr(raw_rapid))
print("EasyOCR:", repr(raw_sec))
print("Clean Rapid:", repr(post_process_latin_ocr(raw_rapid)[0]))
print("Clean Easy:", repr(post_process_latin_ocr(raw_sec)[0]))
