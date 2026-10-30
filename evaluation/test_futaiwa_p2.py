import sys
sys.path.insert(0, ".")
from app import TranslationEngine

engine = TranslationEngine.get_instance()
res = engine.process_single_page(
    'temp/f4c22359/extracted/page_001.png',
    target_lang='ind_Latn',
    translation_backend='nllb',
    debug_dir='evaluation/debug_p2_current'
)
res['images']['ind_Latn'].save('evaluation/futaiwa_p2_translated.png')
print("Saved evaluation/futaiwa_p2_translated.png")
for b in res['debug_dump']['bubbles']:
    print(f"Bubble {b['bubble_id']}: OCR={repr(b['ocr_text'])} -> Trans={repr(b['translation']['ind_Latn'])}")
