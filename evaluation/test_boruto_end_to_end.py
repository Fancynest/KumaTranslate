import sys
sys.path.insert(0, ".")
import json
from app import TranslationEngine

engine = TranslationEngine.get_instance()
img_path = 'temp/bd83f15e/extracted/page_001.jpg'

res = engine.process_single_page(
    img_path=img_path,
    target_lang="ind_Latn",
    translation_backend="nllb",
    series_id="Boruto",
    save_debug=True,
    debug_dir="evaluation/debug_boruto"
)

out_img = res["images"]["ind_Latn"]
out_img.save("evaluation/boruto_perfect_translated.png")
print("Successfully processed Boruto end-to-end!")
print("Saved to evaluation/boruto_perfect_translated.png")

with open("evaluation/debug_boruto/page_001_debug.json", "r", encoding="utf-8") as f:
    dump = json.load(f)

for b in dump.get("bubbles", []):
    print(f"[{b.get('type')}] bbox={b.get('bbox')}")
    print(f"   OCR:   {b.get('ocr_text')}")
    print(f"   TRANS: {b.get('translation')}")
