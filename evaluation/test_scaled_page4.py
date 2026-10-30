import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img = Image.open('temp/9dff84e4/extracted/page_001.png').convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

detector = BubbleDetector()
bubbles = detector.detect(img_scaled)
ocr = MultiLingualOCR()

print(f"Scaled image: {img_scaled.size}, {len(bubbles)} bubbles")
for i, b in enumerate(bubbles):
    bx, by, bw, bh = b['bbox']
    rapid_def, _ = ocr.rapid_ocr(b['crop'])
    r_def_txt = ' '.join([r[1] for r in rapid_def]) if rapid_def else ''
    rapid_en, _ = ocr.rapid_ocr_en(b['crop']) if ocr.rapid_ocr_en else (None, None)
    r_en_txt = ' '.join([r[1] for r in rapid_en]) if rapid_en else ''
    read_txt = ocr.read_text(b['crop'])
    print(f"\nBubble #{i} [bbox={b['bbox']}]:")
    print(f"  rapid_def: {repr(r_def_txt)}")
    print(f"  rapid_en:  {repr(r_en_txt)}")
    print(f"  read_text: {repr(read_txt)}")
