import sys
sys.path.insert(0, '.')
import cv2
from src.ocr import MultiLingualOCR
from src.text_utils import post_process_latin_ocr

ocr = MultiLingualOCR()
b0 = cv2.imread('evaluation/p14_b0.png')
b2 = cv2.imread('evaluation/p14_b2.png')

# Invert b0 and b2
b0_inv = 255 - b0
b2_inv = 255 - b2

raw0 = ocr.read_text(b0_inv, lang_hint="eng_Latn")
clean0, q0 = post_process_latin_ocr(raw0)
print(f"b0 inverted -> raw: {raw0!r}")
print(f"b0 clean: {clean0!r} (quality={q0:.2f})")

raw2 = ocr.read_text(b2_inv, lang_hint="eng_Latn")
clean2, q2 = post_process_latin_ocr(raw2)
print(f"b2 inverted -> raw: {raw2!r}")
print(f"b2 clean: {clean2!r} (quality={q2:.2f})")
