import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

img_scaled = Image.open('temp/80f46fe3/extracted/page_001.png').convert("RGB")
w_orig, h_orig = img_scaled.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
img_scaled = img_scaled.resize((int(w_orig * scale_up), int(h_orig * scale_up)), Image.Resampling.LANCZOS)

# YOLO #2: bbox around (337, 332, 100, 144)
# text is "CAN YOU BE NICER TO HIM?"
crop = img_scaled.crop((337, 332, 337 + 100, 332 + 144))
crop.save('evaluation/crop_yolo2.png')

ocr = MultiLingualOCR()
print("Rapid:", repr(ocr.rapid_ocr(crop)[0]))
print("Rapid EN:", repr(ocr.rapid_ocr_en(crop)[0]))
print("EasyOCR:", repr(ocr.read_text_secondary(crop, lang='en')))
