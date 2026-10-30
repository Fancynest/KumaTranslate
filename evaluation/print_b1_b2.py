import sys
sys.path.insert(0, ".")
import json

with open('evaluation/debug_page4/page_001_debug.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for b in d['bubbles'][:3]:
    print(b['bubble_id'], b['bbox'], b['text_bbox'], b['ocr_text'], '->', b['translation']['ind_Latn'])
