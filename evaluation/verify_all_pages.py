import sys
sys.path.insert(0, ".")
import os
import cv2
import numpy as np
from PIL import Image
from app import TranslationEngine

engine = TranslationEngine.get_instance()

# 1. Test Futaiwa Island page (English)
futaiwa_path = "temp/59c91ad5/extracted/page_001.png"
if os.path.exists(futaiwa_path):
    print("=== Processing Futaiwa Island Page ===")
    res_futaiwa = engine.process_single_page(futaiwa_path, target_lang="ind_Latn", translation_backend="nllb", debug_dir="evaluation/debug_futaiwa")
    out_id = res_futaiwa["images"]["ind_Latn"]
    out_id.save("evaluation/futaiwa_verified_translated.png")
    bubbles = res_futaiwa["debug_dump"]["bubbles"]
    print(f"Saved: evaluation/futaiwa_verified_translated.png ({len(bubbles)} bubbles)")
    for i, b in enumerate(bubbles):
        print(f"Futaiwa Bubble {i}: OCR='{b.get('ocr_text')}' -> Trans='{b.get('translation', {}).get('ind_Latn')}'")

# 2. Test Boruto English Page
boruto_en_path = "temp/bd83f15e/extracted/page_001.jpg"
if os.path.exists(boruto_en_path):
    print("\n=== Processing Boruto English Page ===")
    res_boruto_en = engine.process_single_page(boruto_en_path, target_lang="ind_Latn", translation_backend="nllb", debug_dir="evaluation/debug_boruto")
    out_b_id = res_boruto_en["images"]["ind_Latn"]
    out_b_id.save("evaluation/boruto_en_verified_translated.png")
    bubbles = res_boruto_en["debug_dump"]["bubbles"]
    print(f"Saved: evaluation/boruto_en_verified_translated.png ({len(bubbles)} bubbles)")
    for i, b in enumerate(bubbles):
        print(f"Boruto Bubble {i}: OCR='{b.get('ocr_text')}' -> Trans='{b.get('translation', {}).get('ind_Latn')}'")

# 3. Test Japanese Manga Page (get.jpg)
get_path = "temp/f11356db/get.jpg"
if os.path.exists(get_path):
    print("\n=== Processing Japanese Manga Page (get.jpg) ===")
    res_ja = engine.process_single_page(get_path, target_lang="ind_Latn", translation_backend="nllb", translate_free_text=False, debug_dir="evaluation/debug_get_ja")
    out_ja_id = res_ja["images"]["ind_Latn"]
    out_ja_id.save("evaluation/get_ja_verified_translated.png")
    bubbles = res_ja["debug_dump"]["bubbles"]
    print(f"Saved: evaluation/get_ja_verified_translated.png ({len(bubbles)} bubbles)")
    for i, b in enumerate(bubbles):
        print(f"Japanese Bubble {i}: OCR='{b.get('ocr_text')}' -> Trans='{b.get('translation', {}).get('ind_Latn')}'")
