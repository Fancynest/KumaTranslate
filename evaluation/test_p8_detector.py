import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img = Image.open('temp/ae483818/extracted/page_0008.png').convert("RGB")
detector = BubbleDetector()
bubbles = detector.detect(img)
ocr = MultiLingualOCR()

print(f"Total bubbles detected by detector.detect(): {len(bubbles)}")
for i, b in enumerate(bubbles):
    txt = ocr.read_text(b["crop"], lang_hint="eng_Latn")
    print(f"#{i} [{b.get('type')}] bbox={b['bbox']} text_bbox={b.get('text_bbox')} ocr={repr(txt)}")
