import sys
sys.path.insert(0, '.')
import cv2
import json
import numpy as np
from PIL import Image
from app import TranslationEngine

pipeline = TranslationEngine.get_instance()
img_path = 'temp/ae483818/extracted/page_0014.png'

res = pipeline.process_single_page(img_path, doc_lang_hint='eng_Latn')
print(f"Detected lang: {res.get('detected_lang')}")
dump = res.get('debug_dump', {})
print(f"Total bubbles in dump: {dump.get('total_bubbles')}")

for i, b in enumerate(dump.get('bubbles', [])):
    print(f"Bubble {i} ({b.get('bbox')}):")
    print(f"  raw_ocr:     {b.get('raw_ocr')!r}")
    print(f"  clean_ocr:   {b.get('ocr_text')!r} (quality={b.get('ocr_quality')})")
    print(f"  translation: {b.get('translation')!r}")

# Save translated image
out_img = res['images']['ind_Latn']
out_img.save('evaluation/p14_translated_result.png')

# Save crop of Bubble 0 (black bubble) and Bubble 1 (black bubble) in translated image
out_cv = cv2.imread('evaluation/p14_translated_result.png')
orig_cv = cv2.imread(img_path)

# Let's crop bubble 0
# Bbox in dump:
for i, b in enumerate(dump.get('bubbles', [])):
    bx, by, bw, bh = b.get('bbox')
    pad = 10
    y1, y2 = max(0, by - pad), min(out_cv.shape[0], by + bh + pad)
    x1, x2 = max(0, bx - pad), min(out_cv.shape[1], bx + bw + pad)
    c_out = out_cv[y1:y2, x1:x2]
    c_orig = orig_cv[y1:y2, x1:x2]
    cv2.imwrite(f'evaluation/p14_bubble_{i}_rendered.png', c_out)
    cv2.imwrite(f'evaluation/p14_bubble_{i}_orig.png', c_orig)
    print(f"Saved bubble {i} crops. Out mean: {np.mean(c_out):.1f}, Orig mean: {np.mean(c_orig):.1f}")
