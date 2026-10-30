import sys
sys.path.insert(0, ".")
import cv2
import json
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img_path = 'temp/9dff84e4/extracted/page_001.png'
detector = BubbleDetector()
bubbles = detector.detect(img_path)
ocr = MultiLingualOCR()

print(f"Detected {len(bubbles)} bubbles:")
for i, b in enumerate(bubbles):
    bx, by, bw, bh = b['bbox']
    rapid_res, _ = ocr.rapid_ocr(b['crop'])
    rapid_txt = ' '.join([r[1] for r in rapid_res]) if rapid_res else ''
    rapid_en_res, _ = ocr.rapid_ocr_en(b['crop']) if ocr.rapid_ocr_en else (None, None)
    rapid_en_txt = ' '.join([r[1] for r in rapid_en_res]) if rapid_en_res else ''
    full_read = ocr.read_text(b['crop'])
    sec_read = ocr.read_text_secondary(b['crop'], lang='en')
    print(f"\nBubble #{i} [bbox={b['bbox']}]:")
    print(f"   rapid_def: {repr(rapid_txt)}")
    print(f"   rapid_en:  {repr(rapid_en_txt)}")
    print(f"   read_text: {repr(full_read)}")
    print(f"   sec_easy:  {repr(sec_read)}")
