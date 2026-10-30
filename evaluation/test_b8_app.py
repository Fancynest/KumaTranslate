import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

img = Image.open("temp/bd83f15e/extracted/page_001.jpg")
detector = BubbleDetector()
ocr = MultiLingualOCR()

bubbles = detector.detect(img)
for i, b in enumerate(bubbles):
    if b["bbox"][0] in [46, 25] and b["bbox"][1] in [673, 674, 372]:
        print(f"Index {i}, bbox={b['bbox']}:")
        raw_none = ocr.read_text(b["crop"], lang_hint=None)
        print(f"  raw_none: '{raw_none}'")
        raw_en = ocr.read_text(b["crop"], lang_hint="eng_Latn")
        print(f"  raw_en:   '{raw_en}'")
        clean_none, q_none = post_process_latin_ocr(raw_none)
        print(f"  clean_none: '{clean_none}', q={q_none}")
        clean_en, q_en = post_process_latin_ocr(raw_en)
        print(f"  clean_en:   '{clean_en}', q={q_en}")
