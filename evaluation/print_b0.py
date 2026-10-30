import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector

img = Image.open("temp/59c91ad5/extracted/page_001.png")
detector = BubbleDetector()
bubbles = detector.detect(img)
b0 = bubbles[0]
for k, v in b0.items():
    if k != 'bubble_mask':
        print(f"{k}: {v}")
