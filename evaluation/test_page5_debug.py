import sys
sys.path.insert(0, '.')
import cv2
import json
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.translator import MangaTranslator
from src.text_utils import post_process_latin_ocr

img_path = 'temp/ae483818/extracted/page_0005.png'
detector = BubbleDetector()
ocr = MultiLingualOCR()
translator = MangaTranslator()

bubbles = detector.detect(img_path)
print(f"Total detected bubbles: {len(bubbles)}")

for i, b in enumerate(bubbles):
    crop = b["crop"]
    raw = ocr.read_text(crop, lang_hint="eng_Latn")
    t_clean, q = post_process_latin_ocr(raw)
    trans = translator.translate_batch([t_clean], src_lang="eng_Latn", tgt_lang="ind_Latn")[0]
    bx, by, bw, bh = b["bbox"]
    print(f"--- Bubble {i} ({bx},{by},{bw},{bh}) ---")
    print(f"  RAW:   {raw!r}")
    print(f"  CLEAN: {t_clean!r} (quality={q:.2f})")
    print(f"  TRANS: {trans!r}")
