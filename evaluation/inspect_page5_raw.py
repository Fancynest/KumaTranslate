import sys
sys.path.insert(0, '.')
import cv2
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.translator import MangaTranslator

img_path = 'temp/ae483818/extracted/page_0005.png'
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img_path)
print(f'Total detected: {len(bubbles)}')
for i, b in enumerate(bubbles):
    bx, by, bw, bh = b['bbox']
    text, q = ocr.extract_text(img_path, b)
    src = b.get('source')
    btype = b.get('type')
    print(f"Bubble {i}: bbox=({bx},{by},{bw},{bh}), source={src}, type={btype}")
    print(f"  OCR: {text!r} (quality={q:.2f})")
