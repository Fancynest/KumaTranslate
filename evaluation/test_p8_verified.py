import os
import sys
import json
sys.path.insert(0, '.')
from app import TranslationEngine

engine = TranslationEngine.get_instance()
res = engine.process_single_page(
    'temp/ae483818/extracted/page_0008.png',
    target_lang='ind_Latn',
    doc_lang_hint='eng_Latn',
    save_debug=True,
    debug_dir='evaluation/debug_p8_verified',
    translate_free_text=False
)

print(f"Total bubbles: {len(res['bubbles'])}")
for i, b in enumerate(res['bubbles']):
    print(f"#{i} [{b.get('type')}] bbox={b.get('bbox')}")
    print(f"   OCR:   {b.get('ocr_text')}")
    print(f"   TRANS: {b.get('translation')}")

if 'ind_Latn' in res['images']:
    res['images']['ind_Latn'].save('evaluation/p8_verified_translated.png')
    print("Saved evaluation/p8_verified_translated.png")
