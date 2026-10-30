import json
import sys

with open("temp/f11356db/debug/page_001_debug.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, b in enumerate(data.get("bubbles", [])):
    sys.stdout.buffer.write(f"Bubble {i}:\n".encode("utf-8"))
    sys.stdout.buffer.write(f"  OCR: {b.get('ocr_text')}\n".encode("utf-8"))
    sys.stdout.buffer.write(f"  Trans: {b.get('translation')}\n".encode("utf-8"))
    sys.stdout.buffer.write(f"  Bbox: {b.get('bbox')}\n".encode("utf-8"))
