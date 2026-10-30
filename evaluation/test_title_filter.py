import sys
sys.path.insert(0, '.')
from PIL import Image
import numpy as np
from src.detector import BubbleDetector

det = BubbleDetector()

for p in ['temp/ae483818/extracted/page_0001.png', 'temp/ae483818/extracted/page_0002.png', 'temp/ae483818/extracted/page_0008.png']:
    im = Image.open(p)
    w_orig, h_orig = im.size
    print(f'=== Testing {p} ({w_orig}x{h_orig}) ===')
    # Simulate updated detector
    # Let's inspect what clusters are detected
    bubbles = det.detect(im, include_free_text=True)
    for i, b in enumerate(bubbles):
        src = b.get("source")
        btype = b.get("type")
        bbox = b.get("bbox")
        print(f"  #{i} [{btype}] src={src} bbox={bbox}")
