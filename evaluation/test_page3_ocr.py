import sys
sys.path.insert(0, '.')
import cv2
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

img_path = 'temp/ae483818/extracted/page_0003.png'
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img_path)
print(f"Total bubbles: {len(bubbles)}")
for i, b in enumerate(bubbles):
    crop = b["crop"]
    raw = ocr.read_text(crop, lang_hint="eng_Latn")
    clean, q = post_process_latin_ocr(raw)
    print(f"Bubble {i} {b['bbox']}:")
    print(f"  RAW:   {raw!r}")
    print(f"  CLEAN: {clean!r} (q={q:.2f})")
