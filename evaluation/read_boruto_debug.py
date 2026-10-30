import json

with open("temp/bd83f15e/debug/page_001_debug.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, b in enumerate(data.get("bubbles", [])):
    print(f"Bubble {i}:")
    print(f"  OCR: {b.get('ocr_text')}")
    print(f"  Trans: {b.get('translation')}")
    print(f"  Bbox: {b.get('bbox')}")
