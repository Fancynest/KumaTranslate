import json

p = 'page_0009_debug.json'
print('============================== ' + p + ' ==============================')
with open('temp/ae483818/debug/' + p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Total bubbles: {len(data.get('bubbles', []))}")
for i, b in enumerate(data.get('bubbles', [])):
    print(f"#{i} [{b.get('type')}] bbox={b.get('bbox')}")
    print(f"   OCR:   {b.get('ocr_text')}")
    print(f"   TRANS: {b.get('translation')}")
