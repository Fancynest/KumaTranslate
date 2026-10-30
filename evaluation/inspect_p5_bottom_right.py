import sys
sys.path.insert(0, '.')
import cv2
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

img_path = 'temp/ae483818/extracted/page_0005.png'
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img_path)
for b in bubbles:
    bx, by, bw, bh = b["bbox"]
    if bx > 900 and by > 1400:
        raw = ocr.read_text(b["crop"], lang_hint="eng_Latn")
        clean, q = post_process_latin_ocr(raw)
        print(f"Bottom-right bubble at ({bx},{by},{bw},{bh}):")
        print(f"  RAW:   {raw!r}")
        print(f"  CLEAN: {clean!r} (q={q:.2f})")
