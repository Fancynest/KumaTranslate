import sys
sys.path.insert(0, ".")
from app import TranslationEngine

engine = TranslationEngine.get_instance()
res = engine.process_single_page(
    'temp/80f46fe3/extracted/page_001.png',
    target_lang='ind_Latn',
    translation_backend='nllb',
    debug_dir='evaluation/debug_p2_check'
)
res['images']['ind_Latn'].save('evaluation/p2_translated_current.png')
print("Saved evaluation/p2_translated_current.png")
for b in res['debug_dump']['bubbles'][:4]:
    print(f"Bubble {b['bubble_id']}: OCR={repr(b['ocr_text'])} -> Trans={repr(b['translation']['ind_Latn'])}")
