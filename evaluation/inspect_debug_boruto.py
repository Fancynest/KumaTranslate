import json

with open("evaluation/debug_boruto/page_001_debug.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, b in enumerate(data.get("bubbles", [])):
    print(f"Bubble {i} bbox={b.get('bbox')}:")
    print(f"  OCR: '{b.get('ocr_text')}'")
    print(f"  Trans: '{b.get('translation', {}).get('ind_Latn')}'")
