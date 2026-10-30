import sys
sys.path.insert(0, ".")
from app import TranslationEngine

engine = TranslationEngine.get_instance()
res = engine.process_single_page(
    'temp/9dff84e4/extracted/page_001.png',
    target_lang='ind_Latn',
    translation_backend='nllb',
    debug_dir='evaluation/debug_page4'
)
res['images']['ind_Latn'].save('evaluation/page4_translated_test.png')
print('Done. Saved evaluation/page4_translated_test.png')
for b in res['debug_dump']['bubbles']:
    print(f"Bubble {b['bubble_id']} [bbox={b['bbox']}]: OCR={repr(b['ocr_text'])} -> Trans={repr(b['translation']['ind_Latn'])}")
