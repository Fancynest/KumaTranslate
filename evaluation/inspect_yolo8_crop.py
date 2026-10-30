import sys
sys.path.insert(0, ".")
from PIL import Image
from src.ocr import MultiLingualOCR

img_scaled = Image.open('temp/80f46fe3/extracted/page_001.png').convert("RGB")
w_orig, h_orig = img_scaled.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
img_scaled = img_scaled.resize((int(w_orig * scale_up), int(h_orig * scale_up)), Image.Resampling.LANCZOS)

# YOLO #8: (79, 9, 110, 135)
crop = img_scaled.crop((79, 9, 79 + 110, 9 + 135))
ocr = MultiLingualOCR()
print("Rapid:", repr(ocr.rapid_ocr(crop)[0]))
print("Rapid EN:", repr(ocr.rapid_ocr_en(crop)[0]))
print("EasyOCR:", repr(ocr.read_text_secondary(crop, lang='en')))
