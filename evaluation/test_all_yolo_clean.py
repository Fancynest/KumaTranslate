import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR

img_path = 'temp/80f46fe3/extracted/page_001.png'
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
img_rgb = np.array(img_scaled)
gray_np = np.array(img_scaled.convert("L"))

detector = BubbleDetector()
raw_text_boxes = detector._detect_raw_boxes(img_rgb)
yolo_detections = detector._detect_yolo_bubbles(img_rgb)
yolo_bubbles = [d[:4] for d in yolo_detections if d[4] == "text_bubble"]

bubble_texts = [[] for _ in range(len(yolo_bubbles))]
for tb in raw_text_boxes:
    tx, ty, tw, th = tb
    text_area = tw * th
    best_containment = 0.0
    best_bubble_idx = -1
    for b_idx, yb in enumerate(yolo_bubbles):
        bx, by, bw, bh = yb
        xi1 = max(tx, bx)
        yi1 = max(ty, by)
        xi2 = min(tx + tw, bx + bw)
        yi2 = min(ty + th, by + bh)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        containment = inter / text_area if text_area > 0 else 0.0
        if containment > best_containment:
            best_containment = containment
            best_bubble_idx = b_idx
    if best_containment >= 0.50 and best_bubble_idx >= 0:
        bubble_texts[best_bubble_idx].append(tb)

ocr = MultiLingualOCR()

print("Testing YOLO #4, #2, #9:")
for b_idx in range(len(yolo_bubbles)):
    assigned_texts = bubble_texts[b_idx]
    if not assigned_texts:
        continue
    tx1 = min(t[0] for t in assigned_texts)
    ty1 = min(t[1] for t in assigned_texts)
    tx2 = max(t[0] + t[2] for t in assigned_texts)
    ty2 = max(t[1] + t[3] for t in assigned_texts)
    text_bbox = (tx1, ty1, max(4, tx2 - tx1), max(4, ty2 - ty1))

    yb = yolo_bubbles[b_idx]
    bubble_mask, safe_bbox = detector._compute_bubble_mask(
        gray=gray_np,
        text_bbox=text_bbox,
        roi_bbox=yb,
        font_size=14.0
    )

    cleaned_rgb = img_rgb.copy()
    sx, sy, sw, sh = safe_bbox
    for other_idx, other_boxes in enumerate(bubble_texts):
        if other_idx == b_idx:
            continue
        for otb in other_boxes:
            ox, oy, ow, oh = otb
            xi1 = max(sx, ox)
            yi1 = max(sy, oy)
            xi2 = min(sx + sw, ox + ow)
            yi2 = min(sy + sh, oy + oh)
            if xi2 > xi1 and yi2 > yi1:
                px1 = max(0, ox - 4)
                py1 = max(0, oy - 4)
                px2 = min(new_w, ox + ow + 4)
                py2 = min(new_h, oy + oh + 4)
                cleaned_rgb[py1:py2, px1:px2] = 255

    crop_img = detector._prepare_ocr_crop(
        img_rgb=cleaned_rgb,
        crop_bbox=safe_bbox,
        bubble_mask=bubble_mask,
        font_size=14.0,
        padding=10,
        target_font_height=28.0
    )

    txt = ocr.read_text(crop_img, lang_hint="eng_Latn")
    print(f"YOLO #{b_idx} [yb={yb}]: OCR={repr(txt)}")
