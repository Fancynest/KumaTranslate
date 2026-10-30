import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img = Image.open("temp/bd83f15e/extracted/page_001.jpg")
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img)
for i, b in enumerate(bubbles):
    if b["bbox"][0] in [46, 25] and b["bbox"][1] in [673, 674, 372]:
        raw = ocr.read_text(b["crop"], lang_hint="eng_Latn")
        print(f"Index {i}, crop shape {b['crop'].size}: raw = '{raw}'")
