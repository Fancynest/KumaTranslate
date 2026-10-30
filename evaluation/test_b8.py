import sys
sys.path.insert(0, ".")
from PIL import Image
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

ocr = MultiLingualOCR()
# Let's test post_process_latin_ocr directly
t1, q1 = post_process_latin_ocr("THEY DON'T NEEDTO KNOW THAT!")
print(f"t1: '{t1}', q1: {q1}")

t2, q2 = post_process_latin_ocr("They done need to know that!")
print(f"t2: '{t2}', q2: {q2}")

# Let's inspect what RapidOCR and EasyOCR return on crop of bubble 8
from src.detector import BubbleDetector
img = Image.open("temp/bd83f15e/extracted/page_001.jpg")
detector = BubbleDetector()
bubbles = detector.detect(img)
for i, b in enumerate(bubbles):
    # bubble at (25, 372, 47, 56)
    if abs(b['bbox'][0] - 25) < 5 and abs(b['bbox'][1] - 372) < 5:
        print(f"Bubble #{i} at {b['bbox']}:")
        raw_primary = ocr.read_text(b["crop"], lang_hint="eng_Latn")
        print(f"  Primary RapidOCR: '{raw_primary}'")
        raw_sec = ocr.read_text_secondary(b["crop"], lang="en")
        print(f"  Secondary EasyOCR: '{raw_sec}'")
        clean_p, q_p = post_process_latin_ocr(raw_primary)
        print(f"  Clean primary: '{clean_p}', q={q_p}")
        clean_s, q_s = post_process_latin_ocr(raw_sec)
        print(f"  Clean secondary: '{clean_s}', q={q_s}")
