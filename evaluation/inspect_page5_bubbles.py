import sys
sys.path.insert(0, ".")
import cv2
import json
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img_path = 'temp/80f46fe3/extracted/page_001.png'
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
if min_dim < 650:
    scale_up = 750.0 / min_dim
    new_w = int(w_orig * scale_up)
    new_h = int(h_orig * scale_up)
    img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
else:
    img_scaled = img

detector = BubbleDetector()
bubbles = detector.detect(img_scaled)
ocr = MultiLingualOCR()

print(f"Total detected bubbles: {len(bubbles)}")
for i, b in enumerate(bubbles):
    rapid_res, _ = (ocr.rapid_ocr_en or ocr.rapid_ocr)(b["crop"])
    rapid_txt = " ".join([r[1] for r in rapid_res]) if rapid_res else ""
    print(f"\n--- Bubble #{i} ---")
    print(f"  bbox:      {b['bbox']}")
    print(f"  text_bbox: {b.get('text_bbox')}")
    print(f"  safe_bbox: {b.get('safe_bbox')}")
    print(f"  type:      {b.get('type')}")
    print(f"  rapid_txt: {repr(rapid_txt)}")
