import sys
sys.path.insert(0, '.')
from PIL import Image
from src.detector import BubbleDetector

det = BubbleDetector()

for p in ['temp/ae483818/extracted/page_0001.png', 'temp/ae483818/extracted/page_0002.png']:
    im = Image.open(p)
    w, h = im.size
    print(f'=== {p} ({w}x{h}) ===')
    bubbles = det.detect(im, include_free_text=True)
    for i, b in enumerate(bubbles):
        src = b.get("source")
        btype = b.get("type")
        bbox = b.get("bbox")
        tbbox = b.get("text_bbox")
        print(f"  #{i} source={src} type={btype} bbox={bbox} text_bbox={tbbox}")
