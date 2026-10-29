import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from rapidocr_onnxruntime import RapidOCR

class BubbleDetector:
    def __init__(self):
        """
        Detector Balon Kata Manga & Komik Canggih:
        1. Tiling dinamis untuk gambar dengan aspect ratio tinggi (webtoon/manhwa strip) + NMS.
        2. Persempit filter banner (hanya header ekstrim non-balon).
        3. Penggabungan berbasis ukuran huruf median, bebas efek berantai (non-transitif),
           dan wajib berada di komponen putih terhubung yang sama.
        4. Mask balon kata dihitung via flood fill presisi yang berhenti di garis kontur hitam.
        5. Konversi warna presisi menggunakan RGB2GRAY.
        6. Mode debug visualisasi overlay untuk evaluasi.
        """
        self.engine = RapidOCR()

    def _box_iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _nms_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.40) -> List[Tuple[int, int, int, int]]:
        """
        Non-Maximum Suppression untuk deduplikasi kotak teks dari tile bertumpuk.
        """
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []

        while sorted_boxes:
            curr = sorted_boxes.pop(0)
            kept.append(curr)
            remaining = []
            for other in sorted_boxes:
                iou = self._box_iou(curr, other)
                xi1 = max(curr[0], other[0])
                yi1 = max(curr[1], other[1])
                xi2 = min(curr[0] + curr[2], other[0] + other[2])
                yi2 = min(curr[1] + curr[3], other[1] + other[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                containment = inter / (other[2] * other[3]) if (other[2] * other[3]) > 0 else 0

                if iou < iou_thresh and containment < 0.75:
                    remaining.append(other)
            sorted_boxes = remaining

        return kept

    def _detect_raw_boxes(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Mendeteksi teks menggunakan DBNet. Jika aspect ratio tinggi (>2.0),
        gunakan tiling vertikal dengan overlap dan merge hasil tile pakai NMS.
        """
        h_orig, w_orig = img_rgb.shape[:2]
        aspect_ratio = max(h_orig, w_orig) / max(1, min(h_orig, w_orig))

        # (1) Tiling untuk gambar dengan rasio panjang/lebar tinggi (misal Webtoon strip)
        if aspect_ratio > 2.0 and h_orig > w_orig:
            tile_h = min(h_orig, max(800, int(w_orig * 1.5)))
            overlap = 180
            stride = max(200, tile_h - overlap)

            raw_boxes = []
            y_start = 0
            while y_start < h_orig:
                y_end = min(h_orig, y_start + tile_h)
                tile = img_rgb[y_start:y_end, :]

                # Deteksi pada tile
                th, tw = tile.shape[:2]
                scale = 800.0 / min(tw, th) if min(tw, th) < 650 else (1400.0 / max(tw, th) if max(tw, th) > 1600 else 1.0)
                if scale != 1.0:
                    scaled_tile = cv2.resize(tile, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4)
                else:
                    scaled_tile = tile

                dt_boxes, _ = self.engine.text_det(scaled_tile)
                if dt_boxes is not None and len(dt_boxes) > 0:
                    for b in dt_boxes:
                        xs = [pt[0] / scale for pt in b]
                        ys = [pt[1] / scale for pt in b]
                        bx = int(max(0, min(xs)))
                        by = int(max(0, min(ys))) + y_start
                        bw = int(min(w_orig - bx, max(xs) - min(xs)))
                        bh = int(min(h_orig - by, max(ys) - min(ys)))
                        if bw >= 4 and bh >= 4:
                            raw_boxes.append((bx, by, bw, bh))

                if y_end >= h_orig:
                    break
                y_start += stride

            # Merge hasil tile pakai NMS
            return self._nms_boxes(raw_boxes, iou_thresh=0.40)

        # Gambar standar: proses langsung dengan normalisasi skala aman
        cur_min = min(w_orig, h_orig)
        cur_max = max(w_orig, h_orig)
        if cur_min < 650:
            scale = 750.0 / cur_min
        elif cur_max > 1600:
            scale = 1400.0 / cur_max
        else:
            scale = 1.0

        if scale != 1.0:
            interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
            det_img = cv2.resize(img_rgb, (int(w_orig * scale), int(h_orig * scale)), interpolation=interp)
        else:
            det_img = img_rgb

        dt_boxes, _ = self.engine.text_det(det_img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        raw_boxes = []
        for b in dt_boxes:
            xs = [pt[0] / scale for pt in b]
            ys = [pt[1] / scale for pt in b]
            bx = int(max(0, min(xs)))
            by = int(max(0, min(ys)))
            bw = int(min(w_orig - bx, max(xs) - min(xs)))
            bh = int(min(h_orig - by, max(ys) - min(ys)))
            if bw >= 4 and bh >= 4:
                raw_boxes.append((bx, by, bw, bh))

        return raw_boxes

    def _compute_bubble_mask(
        self,
        gray: np.ndarray,
        cluster_bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        (4) Hitung bubble mask lewat flood fill presisi tinggi.
        Flood fill mencari warna latar putih di dalam balon dan berhenti tepat pada outline hitam (<155).
        """
        h_img, w_img = gray.shape
        bx, by, bw, bh = cluster_bbox

        pad_x = max(10, int(bw * 1.1))
        pad_y = max(8, int(bh * 0.35))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w_img, bx + bw + pad_x)
        y2 = min(h_img, by + bh + pad_y)

        roi_gray = gray[y1:y2, x1:x2]
        rh, rw = roi_gray.shape

        rel_x = bx - x1
        rel_y = by - y1
        text_roi = roi_gray[rel_y:rel_y+bh, rel_x:rel_x+bw]

        # Cari titik awal (seed) berpixel putih di dalam balon
        pts = np.argwhere(text_roi > 190)
        if len(pts) == 0:
            pts = np.argwhere(text_roi > 160)

        if len(pts) == 0:
            # Fallback jika balon gelap
            mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            mask_full[by:by+bh, bx:bx+bw] = 255
            return mask_full, cluster_bbox

        sy, sx = pts[0] + [rel_y, rel_x]

        # Penghalang batas hitam balon (<150)
        barrier = (roi_gray < 150).astype(np.uint8)

        flood_mask = np.zeros((rh + 2, rw + 2), np.uint8)
        flood_mask[1:-1, 1:-1] = barrier
        # Hilangkan barrier pada teks agar floodfill bisa merambat di antara huruf
        flood_mask[rel_y+1:rel_y+bh+1, rel_x+1:rel_x+bw+1] = 0

        # Jalankan flood fill
        cv2.floodFill(roi_gray.copy(), flood_mask, (sx, sy), 255, loDiff=35, upDiff=35, flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

        filled_roi = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8)
        ys, xs = np.where(filled_roi > 0)

        if len(xs) > 0:
            fb_x = x1 + int(xs.min())
            fb_y = y1 + int(ys.min())
            fb_w = int(xs.max() - xs.min() + 1)
            fb_h = int(ys.max() - ys.min() + 1)

            # Cek jika flood fill bocor ke margin terbuka halaman (bukan balon tertutup)
            is_leaked = (fb_w >= rw - 2 and fb_h >= rh - 2) or (fb_w > w_img * 0.70) or (fb_h > h_img * 0.70)
            if is_leaked:
                pad = max(4, int(min(bw, bh) * 0.15))
                tight_x = max(0, bx - pad)
                tight_y = max(0, by - pad)
                tight_w = min(w_img - tight_x, bw + 2 * pad)
                tight_h = min(h_img - tight_y, bh + 2 * pad)
                bubble_bbox = (tight_x, tight_y, tight_w, tight_h)
                mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
                mask_full[tight_y:tight_y+tight_h, tight_x:tight_x+tight_w] = 255
                return mask_full, bubble_bbox

            bubble_bbox = (fb_x, fb_y, max(bw, fb_w), max(bh, fb_h))

            # Rekonstruksi mask global
            mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = filled_roi * 255
            return mask_full, bubble_bbox

        # Fallback
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        mask_full[by:by+bh, bx:bx+bw] = 255
        return mask_full, cluster_bbox

    def detect(
        self,
        image_input,
        debug: bool = False,
        debug_dir: str = "debug_output"
    ) -> List[Dict[str, Any]]:
        """
        Pipeline Deteksi Lengkap:
        - Input: Gambar RGB / path / PIL Image
        - Output: List balon kata dengan bubble_mask & bubble_bbox presisi.
        """
        # (5) Standarisasi konversi: pastikan img_rgb selalu RGB, dan gunakan RGB2GRAY
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        h_orig, w_orig = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Deteksi teks (dengan Tiling + NMS jika aspect ratio tinggi)
        raw_boxes = self._detect_raw_boxes(img_rgb)
        if not raw_boxes:
            return []

        # 2. (2) Persempit filter banner: hanya hapus banner judul raksasa di tepi paling atas (top 8% & width > 45%)
        filtered_boxes = []
        for b in raw_boxes:
            bx, by, bw, bh = b
            # Banner chapter / logo majalah / judul raksasa di tepi atas
            if by < h_orig * 0.08 and bw > w_orig * 0.45:
                continue
            # Kotak raksasa (>35% halaman)
            if (bw * bh) > (w_orig * h_orig * 0.35):
                continue
            filtered_boxes.append(b)

        if not filtered_boxes:
            return []

        # 3. (3) Hitung ukuran huruf median
        box_min_dims = [min(b[2], b[3]) for b in filtered_boxes]
        median_font_size = float(np.median(box_min_dims)) if box_min_dims else 8.0
        max_dist = max(4, int(median_font_size * 1.35))

        # 4. (3) Analisis Komponen Putih Terhubung (Connected Components)
        _, white_bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        num_comps, comp_labels = cv2.connectedComponents(white_bin)

        box_comps = []
        for b in filtered_boxes:
            bx, by, bw, bh = b
            roi_comps = comp_labels[by:by+bh, bx:bx+bw]
            non_zero = roi_comps[roi_comps > 0]
            cid = int(np.bincount(non_zero).argmax()) if len(non_zero) > 0 else 0
            box_comps.append(cid)

        # 5. (3) Adjacency Graph: syaratkan dua box berada di komponen putih yang sama, hapus efek berantai
        n = len(filtered_boxes)
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Syarat komponen putih yang sama (jika terpisah garis border/gutter hitam, tidak boleh gabung!)
                if box_comps[i] > 0 and box_comps[j] > 0 and box_comps[i] != box_comps[j]:
                    continue

                b1, b2 = filtered_boxes[i], filtered_boxes[j]
                dx = max(0, max(b1[0], b2[0]) - min(b1[0] + b1[2], b2[0] + b2[2]))
                dy = max(0, max(b1[1], b2[1]) - min(b1[1] + b1[3], b2[1] + b2[3]))

                can_merge = False
                if dx <= max_dist and dy <= max_dist:
                    can_merge = True
                elif (min(b1[1] + b1[3], b2[1] + b2[3]) - max(b1[1], b2[1])) > 0 and dx <= int(max_dist * 1.5):
                    can_merge = True
                elif (min(b1[0] + b1[2], b2[0] + b2[2]) - max(b1[0], b2[0])) > 0 and dy <= int(max_dist * 1.5):
                    can_merge = True

                if can_merge:
                    adj[i].append(j)
                    adj[j].append(i)

        # Hapus efek berantai: Temukan cluster dan batasi batas maksimal cluster
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            comp = []
            q = [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        # Cegah chaining: periksa jarak langsung ke box pertama dalam cluster
                        first_b = filtered_boxes[comp[0]]
                        neigh_b = filtered_boxes[neighbor]
                        chain_dx = max(0, max(first_b[0], neigh_b[0]) - min(first_b[0] + first_b[2], neigh_b[0] + neigh_b[2]))
                        chain_dy = max(0, max(first_b[1], neigh_b[1]) - min(first_b[1] + first_b[3], neigh_b[1] + neigh_b[3]))
                        if chain_dx <= max_dist * 4 and chain_dy <= max_dist * 6:
                            visited.add(neighbor)
                            q.append(neighbor)

            xs1 = min([filtered_boxes[k][0] for k in comp])
            ys1 = min([filtered_boxes[k][1] for k in comp])
            xs2 = max([filtered_boxes[k][0] + filtered_boxes[k][2] for k in comp])
            ys2 = max([filtered_boxes[k][1] + filtered_boxes[k][3] for k in comp])
            clusters.append((xs1, ys1, xs2 - xs1, ys2 - ys1))

        # 6. Urutkan urutan baca manga (atas ke bawah, kanan ke kiri)
        clusters.sort(key=lambda b: (b[1] // max(30, h_orig // 8), -b[0]))

        # 7. (4) Hitung bubble mask lewat flood fill & pakai bounds-nya untuk inpainter dan renderer
        bubbles = []
        for text_box in clusters:
            bx, by, bw, bh = text_box
            bubble_mask, bubble_bbox = self._compute_bubble_mask(gray, text_box)

            # Crop teks untuk OCR
            pad = max(2, int(min(bw, bh) * 0.15))
            px = max(0, bx - pad)
            py = max(0, by - pad)
            pw = min(w_orig - px, bw + 2 * pad)
            ph = min(h_orig - py, bh + 2 * pad)
            crop_rgb = img_rgb[py:py+ph, px:px+pw]

            bubbles.append({
                "text_bbox": text_box,
                "bubble_bbox": bubble_bbox,
                "bubble_mask": bubble_mask,
                "bbox": bubble_bbox,  # kompatibilitas: default gunakan bubble_bbox!
                "crop": Image.fromarray(crop_rgb),
                "area": bubble_bbox[2] * bubble_bbox[3],
                "median_font_size": median_font_size
            })

        # 8. (6) Mode Debug: Simpan visualisasi overlay jika debug=True
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
            overlay = img_rgb.copy()

            # Layer mask transparan hijau untuk bubble_mask
            mask_color = np.zeros_like(img_rgb)
            for b in bubbles:
                mask_color[b["bubble_mask"] > 0] = [0, 220, 100]

            alpha = 0.28
            overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0)

            # Gambar raw DBNet boxes (Biru)
            for rb in raw_boxes:
                rx, ry, rw, rh = rb
                cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (50, 120, 255), 1)

            # Gambar final bubble bounding boxes (Hijau terang) & Text cluster (Merah)
            for i, b in enumerate(bubbles):
                tx, ty, tw, th = b["text_bbox"]
                cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (255, 60, 60), 1)

                bx, by, bw, bh = b["bubble_bbox"]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"#{i+1} ({bw}x{bh})",
                    (bx, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            debug_filename = "debug_overlay.png"
            if isinstance(image_input, str):
                debug_filename = f"debug_{os.path.basename(image_input)}"
            debug_path = os.path.join(debug_dir, debug_filename)
            Image.fromarray(overlay).save(debug_path)
            print(f"[DEBUG] Overlay deteksi disimpan ke: {debug_path}")

        return bubbles

if __name__ == "__main__":
    detector = BubbleDetector()
    res = detector.detect("evaluation/images/wikipe_page1.jpg", debug=True, debug_dir="debug_output")
    print(f"Total balon terdeteksi: {len(res)}")
    for i, b in enumerate(res):
        print(f"Balon #{i+1}: text_box={b['text_bbox']}, bubble_bbox={b['bubble_bbox']}")
