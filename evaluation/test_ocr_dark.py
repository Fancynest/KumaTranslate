import sys
sys.path.insert(0, '.')
import cv2
from PIL import Image
from src.ocr import MultiLingualOCR

ocr = MultiLingualOCR()
b0 = cv2.imread('evaluation/p14_b0.png')
b2 = cv2.imread('evaluation/p14_b2.png')

print("b0 shape:", b0.shape)
res0 = ocr.read_text(b0, lang_hint="eng_Latn")
print("b0 read_text(eng_Latn):", repr(res0))

res0_auto = ocr.read_text(b0)
print("b0 read_text(auto):", repr(res0_auto))

print("b2 shape:", b2.shape)
res2 = ocr.read_text(b2, lang_hint="eng_Latn")
print("b2 read_text(eng_Latn):", repr(res2))

res2_auto = ocr.read_text(b2)
print("b2 read_text(auto):", repr(res2_auto))
