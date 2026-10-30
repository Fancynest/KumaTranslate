import sys
sys.path.insert(0, ".")
from app import TranslationEngine

engine = TranslationEngine.get_instance()
img_path = 'temp/ae483818/extracted/page_0008.png'

res = engine.process_single_page(
    img_path=img_path,
    target_lang="ind_Latn",
    translation_backend="nllb",
    series_id="Futaiwa",
    save_debug=True,
    debug_dir="evaluation/debug_p8_test"
)

out_img = res["images"]["ind_Latn"]
out_img.save("evaluation/p8_test_translated.png")
print("Saved evaluation/p8_test_translated.png")

import json
with open("evaluation/debug_p8_test/page_0008_debug.json", "r", encoding="utf-8") as f:
    dump = json.load(f)

for b in dump.get("bubbles", []):
    print(f"[{b.get('type')}] bbox={b.get('bbox')}")
    print(f"   OCR:   {b.get('ocr_text')}")
    print(f"   TRANS: {b.get('translation')}")
