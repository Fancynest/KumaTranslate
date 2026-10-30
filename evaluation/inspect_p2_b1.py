import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector

img_path = 'temp/80f46fe3/extracted/page_001.png'
detector = BubbleDetector()
bubbles = detector.detect(img_path)
print(f"Total bubbles: {len(bubbles)}")
for i, b in enumerate(bubbles[:4]):
    print(f"Bubble #{i}: bbox={b['bbox']}, text_bbox={b.get('text_bbox')}, safe_bbox={b.get('safe_bbox')}")
