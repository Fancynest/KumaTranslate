import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img_path = 'temp/ae483818/extracted/page_0014.png'
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img_path)
print(f"Total detected bubbles in p14: {len(bubbles)}")
for i, b in enumerate(bubbles):
    bx, by, bw, bh = b['bbox']
    crop = b['crop']
    crop_np = np.array(crop)
    cv2.imwrite(f'evaluation/p14_bubble_{i}_crop.png', cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR))
    raw = ocr.read_text(crop, lang_hint="eng_Latn")
    print(f"Bubble {i} {b['bbox']}:")
    print(f"  source={b.get('source')}, type={b.get('type')}")
    print(f"  crop mean={np.mean(crop_np):.1f}, min/max={np.min(crop_np)}/{np.max(crop_np)}")
    print(f"  OCR RAW: {raw!r}")
