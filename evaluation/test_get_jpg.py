import os
import sys
sys.path.insert(0, '.')
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.translator import MangaTranslator

det = BubbleDetector()
ocr = MultiLingualOCR()
tr = MangaTranslator()

img = Image.open('temp/d8cee2a7/get.jpg')
bubbles = det.detect(img)
print(f'Detected bubbles: {len(bubbles)}')

raw_texts = []
for i, b in enumerate(bubbles):
    t = ocr.read_text(b['crop'], lang_hint='jpn_Jpan')
    raw_texts.append(t)
    print(f"{i}: [{b['type']}] {t}")

print("\n--- TRANSLATING ---")
trans = tr.translate_batch(raw_texts, src_lang='jpn_Jpan', tgt_lang='ind_Latn')
for i, (orig, t) in enumerate(zip(raw_texts, trans)):
    print(f"{i}: {orig} -> {t}")
