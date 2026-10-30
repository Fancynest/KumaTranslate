import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from src.detector import BubbleDetector

img_path = 'temp/ae483818/extracted/page_0008.png'
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
img_rgb = np.array(img_scaled)

detector = BubbleDetector()
raw_text_boxes = detector._detect_raw_boxes(img_rgb)
yolo_detections = detector._detect_yolo_bubbles(img_rgb)

print(f"Total raw text boxes: {len(raw_text_boxes)}")
print(f"Total YOLO detections: {len(yolo_detections)}")

# Let's see what is detected around x: 100..300, y: 1100..1400 (scaled coordinates)
# In original coordinates: x: 100..400, y: 1100..1500
# On page 7: Makoto is holding the turban shell at x ~ 300, y ~ 1000..1400
# Under "DID YOU KNOW HORNED TURBANS..." there is "OH, I DIDN'T KNOW THAT!"
print("\n--- YOLO detections near middle-left ---")
for yd in yolo_detections:
    # Scale back to orig for inspection
    ox = int(yd[0] / scale_up)
    oy = int(yd[1] / scale_up)
    ow = int(yd[2] / scale_up)
    oh = int(yd[3] / scale_up)
    if 50 <= ox <= 400 and 800 <= oy <= 1600:
        print(f"YOLO: orig=({ox}, {oy}, {ow}, {oh}) scaled={yd[:4]} cls={yd[4]} conf={yd[5]:.2f}")

print("\n--- Raw DBNet boxes near middle-left ---")
for tb in raw_text_boxes:
    ox = int(tb[0] / scale_up)
    oy = int(tb[1] / scale_up)
    ow = int(tb[2] / scale_up)
    oh = int(tb[3] / scale_up)
    if 50 <= ox <= 400 and 800 <= oy <= 1600:
        print(f"DBNet: orig=({ox}, {oy}, {ow}, {oh}) scaled={tb}")
