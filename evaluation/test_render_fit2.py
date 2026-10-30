from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (600, 800), (255, 255, 255))
draw = ImageDraw.Draw(img)

text = "Apakah tidak mengantuk bangun begitu awal?"
bbox = (223, 484, 47, 75)
bx, by, bw, bh = bbox
pad_x = max(2, int(bw * 0.08))
pad_y = max(2, int(bh * 0.08))
rect_w = max(8, bw - 2 * pad_x)
rect_h = max(8, bh - 2 * pad_y)

words = text.split()
font_path = "C:/Windows/Fonts/comicbd.ttf"

for sz in range(24, 7, -1):
    f = ImageFont.truetype(font_path, sz)
    sample_bbox = draw.textbbox((0, 0), "Ag", font=f)
    line_h = sample_bbox[3] - sample_bbox[1]
    spacing = max(1, int(sz * 0.12))
    
    overflow = False
    for w in words:
        w_bb = draw.textbbox((0, 0), w, font=f)
        if (w_bb[2] - w_bb[0]) > rect_w:
            overflow = True
            break
    if overflow:
        continue
        
    lines = []
    cur = []
    for w in words:
        cand = " ".join(cur + [w])
        cand_bb = draw.textbbox((0, 0), cand, font=f)
        if (cand_bb[2] - cand_bb[0]) <= rect_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                lines.append(w)
                cur = []
    if cur:
        lines.append(" ".join(cur))
        
    tot_h = len(lines) * line_h + (len(lines) - 1) * spacing
    print(f"Size {sz}: fits width! tot_h={tot_h}, rect_h={rect_h}, num_lines={len(lines)}")
    if tot_h <= rect_h:
        print(f"PERFECT FIT at size {sz}:")
        for l in lines:
            print("  ", l)
        break
