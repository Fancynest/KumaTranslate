import json

with open("evaluation/debug_boruto/page_001_debug.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for b in data.get("bubbles", []):
    if "need" in b.get("ocr_text", "").lower():
        print(json.dumps(b, indent=2))
