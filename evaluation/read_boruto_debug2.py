import json

with open("temp/bd83f15e/debug/page_001_debug.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, b in enumerate(data.get("bubbles", [])):
    if i >= 8:
        print(f"Bubble {i}:")
        print("  OCR:", ascii(b.get("ocr_text")))
        print("  Trans:", ascii(b.get("translation")))
        print("  Bbox:", b.get("bbox"))
