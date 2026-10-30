import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.translator import MangaTranslator
from src.renderer import MangaRenderer

img_path = 'temp/80f46fe3/extracted/page_001.png'
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size
min_dim = min(w_orig, h_orig)
scale_up = 750.0 / min_dim
new_w = int(w_orig * scale_up)
new_h = int(h_orig * scale_up)
img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
img_rgb = np.array(img_scaled)
gray = np.array(img_scaled.convert("L"))

detector = BubbleDetector()
raw_text_boxes = detector._detect_raw_boxes(img_rgb)
yolo_detections = detector._detect_yolo_bubbles(img_rgb)
yolo_bubbles = [d[:4] for d in yolo_detections if d[4] == "text_bubble"]

# Step 4: Strict assignment
bubble_texts = [[] for _ in range(len(yolo_bubbles))]
unassigned = []
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
    else:
        unassigned.append(tb)

# Build bubbles
bubbles = []
for b_idx, yb in enumerate(yolo_bubbles):
    bx, by, bw, bh = yb
    assigned_texts = list(bubble_texts[b_idx])
    
    # If no texts were assigned, fallback to crop_dt_boxes
    if not assigned_texts:
        bubble_crop = img_rgb[by:by+bh, bx:bx+bw]
        crop_dt_boxes, _ = detector.engine.text_det(bubble_crop)
        if crop_dt_boxes is not None and len(crop_dt_boxes) > 0:
            for b in crop_dt_boxes:
                xs = [pt[0] for pt in b]
                ys = [pt[1] for pt in b]
                cbx = int(max(0, min(xs)))
                cby = int(max(0, min(ys)))
                cbw = int(max(xs) - min(xs))
                cbh = int(max(ys) - min(ys))
                if cbw >= 4 and cbh >= 4:
                    assigned_texts.append((bx + cbx, by + cby, cbw, cbh))

    if assigned_texts:
        tx1 = min(t[0] for t in assigned_texts)
        ty1 = min(t[1] for t in assigned_texts)
        tx2 = max(t[0] + t[2] for t in assigned_texts)
        ty2 = max(t[1] + t[3] for t in assigned_texts)
        text_bbox = (tx1, ty1, max(4, tx2 - tx1), max(4, ty2 - ty1))
    else:
        pad_w = int(bw * 0.15)
        pad_h = int(bh * 0.15)
        text_bbox = (bx + pad_w, by + pad_h, max(4, bw - 2 * pad_w), max(4, bh - 2 * pad_h))

    bubble_mask, safe_bbox = detector._compute_bubble_mask(
        gray=gray,
        text_bbox=text_bbox,
        roi_bbox=yb,
        font_size=14.0
    )

    # Clean foreign text boxes for OCR crop
    cleaned_rgb = img_rgb.copy()
    sx, sy, sw, sh = safe_bbox
    for other_idx, other_boxes in enumerate(bubble_texts):
        if other_idx == b_idx:
            continue
        for otb in other_boxes:
            ox, oy, ow, oh = otb
            if max(sx, ox) < min(sx + sw, ox + ow) and max(sy, oy) < min(sy + sh, oy + oh):
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

    bubbles.append({
        "id": b_idx,
        "text_bbox": text_bbox,
        "bubble_bbox": yb,
        "safe_bbox": safe_bbox,
        "bubble_mask": bubble_mask,
        "bbox": safe_bbox,
        "crop": crop_img,
        "source": "yolo",
        "type": "text_bubble"
    })

# Now resolve safe_bbox and mask overlap between connected bubbles!
for i in range(len(bubbles)):
    for j in range(i + 1, len(bubbles)):
        b1 = bubbles[i]
        b2 = bubbles[j]
        s1 = b1["safe_bbox"]
        s2 = b2["safe_bbox"]
        
        # Check overlap
        xi1 = max(s1[0], s2[0])
        yi1 = max(s1[1], s2[1])
        xi2 = min(s1[0] + s1[2], s2[0] + s2[2])
        yi2 = min(s1[1] + s1[3], s2[1] + s2[3])
        
        if xi2 > xi1 and yi2 > yi1:
            # Overlap detected!
            t1 = b1["text_bbox"]
            t2 = b2["text_bbox"]
            c1x, c1y = t1[0] + t1[2] / 2.0, t1[1] + t1[3] / 2.0
            c2x, c2y = t2[0] + t2[2] / 2.0, t2[1] + t2[3] / 2.0
            
            dx = abs(c1x - c2x)
            dy = abs(c1y - c2y)
            
            print(f"Resolving collision between Bubble #{i} and #{j}: dx={dx:.1f}, dy={dy:.1f}")
            if dx >= dy:
                # Primarily separated horizontally
                if c1x < c2x:
                    left_b, right_b = b1, b2
                else:
                    left_b, right_b = b2, b1
                
                # Split x between right edge of left text and left edge of right text
                lt = left_b["text_bbox"]
                rt = right_b["text_bbox"]
                split_x = int(( (lt[0] + lt[2]) + rt[0] ) / 2.0)
                
                # Clip left bubble
                ls = left_b["safe_bbox"]
                new_lw = max(lt[2] + 4, split_x - ls[0])
                left_b["safe_bbox"] = (ls[0], ls[1], new_lw, ls[3])
                left_b["bbox"] = left_b["safe_bbox"]
                left_b["bubble_mask"][:, split_x:] = 0
                
                # Clip right bubble
                rs = right_b["safe_bbox"]
                new_rx = max(rs[0], split_x)
                new_rw = max(rt[2] + 4, (rs[0] + rs[2]) - new_rx)
                right_b["safe_bbox"] = (new_rx, rs[1], new_rw, rs[3])
                right_b["bbox"] = right_b["safe_bbox"]
                right_b["bubble_mask"][:, :split_x] = 0
            else:
                # Primarily separated vertically
                if c1y < c2y:
                    top_b, bottom_b = b1, b2
                else:
                    top_b, bottom_b = b2, b1
                
                tt = top_b["text_bbox"]
                bt = bottom_b["text_bbox"]
                split_y = int(( (tt[1] + tt[3]) + bt[1] ) / 2.0)
                
                # Clip top bubble
                ts = top_b["safe_bbox"]
                new_th = max(tt[3] + 4, split_y - ts[1])
                top_b["safe_bbox"] = (ts[0], ts[1], ts[2], new_th)
                top_b["bbox"] = top_b["safe_bbox"]
                top_b["bubble_mask"][split_y:, :] = 0
                
                # Clip bottom bubble
                bs = bottom_b["safe_bbox"]
                new_by = max(bs[1], split_y)
                new_bh = max(bt[3] + 4, (bs[1] + bs[3]) - new_by)
                bottom_b["safe_bbox"] = (bs[0], new_by, bs[2], new_bh)
                bottom_b["bbox"] = bottom_b["safe_bbox"]
                bottom_b["bubble_mask"][:split_y, :] = 0

ocr = MultiLingualOCR()
for idx in [12, 5]:
    b = bubbles[idx]
    txt = ocr.read_text(b["crop"], lang_hint="eng_Latn")
    print(f"Bubble #{idx} safe_bbox={b['safe_bbox']} OCR={repr(txt)}")

from src.inpainter import MangaInpainter
from src.text_utils import postprocess_nllb_indonesian

# Translate
translator = MangaTranslator()
t12 = translator.translate("DON'T SAY THAT!", src_lang="eng_Latn", tgt_lang="ind_Latn")
t12 = postprocess_nllb_indonesian(t12, "DON'T SAY THAT!")
t5 = translator.translate("NOT EVERYDAY WE HAVE PEOPLE CAME ALL THE WAY FROM TOKYO TO LIVE HERE.", src_lang="eng_Latn", tgt_lang="ind_Latn")
t5 = postprocess_nllb_indonesian(t5, "NOT EVERYDAY WE HAVE PEOPLE CAME ALL THE WAY FROM TOKYO TO LIVE HERE.")

bubbles[12]["translation"] = t12
bubbles[5]["translation"] = t5

inpainter = MangaInpainter()
test_bubbles = [bubbles[12], bubbles[5]]
text_bboxes = [b["text_bbox"] for b in test_bubbles]
bubble_masks = [b.get("bubble_mask") for b in test_bubbles]
safe_bboxes = [b.get("safe_bbox") or b["text_bbox"] for b in test_bubbles]

inpainter = MangaInpainter()
clean_img = inpainter.clean_bubbles(img_scaled, text_bboxes, masks=bubble_masks, safe_bboxes=safe_bboxes)

renderer = MangaRenderer()
items_id = [
    {
        "bbox": b.get("safe_bbox") or b["text_bbox"],
        "text": b["translation"],
        "type": b.get("type", "text_bubble"),
        "bubble_mask": b.get("bubble_mask")
    }
    for b in test_bubbles
]

out_img = renderer.render_page_bubbles(clean_img, items_id)
out_img.save("evaluation/test_rendered_bottom_right.png")
print("Saved rendered bottom right image to evaluation/test_rendered_bottom_right.png")
print("T12:", t12)
print("T5:", t5)
