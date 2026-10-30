import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img = Image.open("temp/bd83f15e/extracted/page_001.jpg")
detector = BubbleDetector()
bubbles = detector.detect(img)
ocr = MultiLingualOCR()

print(f"Detected {len(bubbles)} bubbles on Boruto page.")
for i, b in enumerate(bubbles):
    crop = b["crop"]
    # Test with lang_hint="eng_Latn"
    txt_en = ocr.read_text(crop, lang_hint="eng_Latn")
    # Test with default/auto
    txt_auto = ocr.read_text(crop)
    print(f"Bubble {i} (bbox={b['bbox']}):")
    print(f"  eng_Latn: '{txt_en}'")
    print(f"  auto:     '{txt_auto}'")
