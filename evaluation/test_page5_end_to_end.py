import sys
sys.path.insert(0, ".")
import os
import json
from app import TranslationEngine

engine = TranslationEngine.get_instance()
img_path = 'temp/80f46fe3/extracted/page_001.png'

res = engine.process_single_page(
    img_path=img_path,
    target_lang="ind_Latn",
    translation_backend="nllb",
    series_id="Futaiwa",
    save_debug=True,
    debug_dir="evaluation/debug_page5"
)

out_img = res["images"]["ind_Latn"]
out_img.save("evaluation/page5_perfect_translated.png")
print("Successfully processed Page 5 end-to-end!")
print("Saved to evaluation/page5_perfect_translated.png")

with open("evaluation/debug_page5/page_001_debug.json", "r", encoding="utf-8") as f:
    dump = json.load(f)

for b in dump.get("bubbles", []):
    print(f"[{b.get('type')}] bbox={b.get('bbox')}")
    print(f"   OCR:   {b.get('ocr_text')}")
    print(f"   TRANS: {b.get('translation')}")
