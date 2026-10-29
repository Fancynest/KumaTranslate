import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from rapidocr_onnxruntime import RapidOCR

class BubbleDetector:
    def __init__(self, mode: str = "auto"):
        """
        Detector Balon Kata Manga & Komik Multi-Mode:
        - mode: "auto" (coba YOLOv8, fallback heuristik jika gagal),
                "yolo" (hanya YOLOv8 model),
                "heuristic" (heuristik berbasis RapidOCR + flood fill).
        """
        self.mode = mode.lower()
        self.engine = RapidOCR()
        self._yolo_model = None
        self._yolo_init_attempted = False

    def _get_yolo_model(self):
        """
        Lazy-load model YOLOv8 comic-speech-bubble-detector dari Hugging Face.
        Lisensi bobot: Apache-2.0 (ogkalu/comic-speech-bubble-detector-yolov8m).
        """
        if self._yolo_init_attempted:
            return self._yolo_model

        self._yolo_init_attempted = True
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
                filename="comic-speech-bubble-detector.pt"
            )
            self._yolo_model = YOLO(model_path)
            print(f"[YOLO] Model comic-speech-bubble-detector berhasil dimuat. Kelas: {self._yolo_model.names}")
        except Exception as e:
            print(f"[WARN] Gagal memuat YOLOv8 bubble detector: {e}. Menggunakan fallback detektor heuristik.")
            self._yolo_model = None

        return self._yolo_model

    def _box_iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = b1[:4]
        x2, y2, w2, h2 = b2[:4]
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
        Non-Maximum Suppression untuk deduplikasi kotak dari tile bertumpuk.
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
        Mendeteksi teks menggunakan DBNet (RapidOCR). Jika aspect ratio tinggi (>2.0),
        gunakan tiling vertikal dengan overlap dan merge hasil tile pakai NMS.
        """
        h_orig, w_orig = img_rgb.shape[:2]
        aspect_ratio = max(h_orig, w_orig) / max(1, min(h_orig, w_orig))

        if aspect_ratio > 2.0 and h_orig > w_orig:
            tile_h = min(h_orig, max(800, int(w_orig * 1.5)))
            overlap = 180
            stride = max(200, tile_h - overlap)

            raw_boxes = []
            y_start = 0
            while y_start < h_orig:
                y_end = min(h_orig, y_start + tile_h)
                tile = img_rgb[y_start:y_end, :]

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

            return self._nms_boxes(raw_boxes, iou_thresh=0.40)

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

    def _detect_yolo_bubbles(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Deteksi kotak balon kata menggunakan YOLOv8.
        imgsz=1024, conf=0.25, iou=0.5.
        Untuk gambar tinggi (webtoon, aspect ratio > 2.0), gunakan tiling vertikal + NMS
        sesuai hasil benchmark empiris (Mean IoU 0.9427 vs 0.9076).
        """
        model = self._get_yolo_model()
        if model is None:
            return []

        h_orig, w_orig = img_rgb.shape[:2]
        aspect_ratio = max(h_orig, w_orig) / max(1, min(h_orig, w_orig))

        # Jika gambar webtoon panjang: tiling vertikal + NMS
        if aspect_ratio > 2.0 and h_orig > w_orig:
            tile_h = min(h_orig, max(800, int(w_orig * 1.5)))
            overlap = 200
            stride = max(200, tile_h - overlap)

            tiled_candidates = []
            y_start = 0
            while y_start < h_orig:
                y_end = min(h_orig, y_start + tile_h)
                tile = img_rgb[y_start:y_end, :]

                results = model.predict(tile, imgsz=1024, conf=0.25, iou=0.5, verbose=False)
                for b in results[0].boxes:
                    cls_id = int(b.cls[0])
                    cls_name = model.names.get(cls_id, "text_bubble")
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    tiled_candidates.append([x1, y1 + y_start, x2 - x1, y2 - y1, cls_name, conf])

                if y_end >= h_orig:
                    break
                y_start += stride

            # NMS pada kotak hasil tile
            sorted_t = sorted(tiled_candidates, key=lambda b: b[5], reverse=True)
            kept_t = []
            while sorted_t:
                curr = sorted_t.pop(0)
                kept_t.append(tuple(curr))
                sorted_t = [b for b in sorted_t if self._box_iou(curr[:4], b[:4]) < 0.40]

            return kept_t

        # Gambar standar: prediksi langsung
        results = model.predict(img_rgb, imgsz=1024, conf=0.25, iou=0.5, verbose=False)
        bubble_boxes = []
        for b in results[0].boxes:
            cls_id = int(b.cls[0])
            cls_name = model.names.get(cls_id, "text_bubble")
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bw = max(4, x2 - x1)
            bh = max(4, y2 - y1)
            bubble_boxes.append((x1, y1, bw, bh, cls_name, conf))

        return bubble_boxes

    def _compute_bubble_mask(
        self,
        gray: np.ndarray,
        cluster_bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Hitung bubble mask lewat flood fill presisi tinggi.
        Flood fill mencari warna latar putih di dalam balon dan berhenti tepat pada outline hitam (<150).
        """
        h_img, w_img = gray.shape
        bx, by, bw, bh = cluster_bbox

        pad_x = max(10, int(bw * 0.40))
        pad_y = max(8, int(bh * 0.25))
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

            # Cek jika flood fill bocor ke margin terbuka halaman
            is_leaked = (fb_w >= rw - 2 and fb_h >= rh - 2) or (fb_w > w_img * 0.70) or (fb_h > h_img * 0.70)
            if is_leaked:
                pad = max(4, int(min(bw, bh) * 0.15))
                tight_x = max(0, bx - pad)
                tight_y = max(0, by - pad)
                tight_w = min(w_img - tight_x, bw + 2 * pad)
                tight_h = min(h_img - tight_y, bh + 2 * pad)
                safe_bbox = (tight_x, tight_y, tight_w, tight_h)
                mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
                mask_full[tight_y:tight_y+tight_h, tight_x:tight_x+tight_w] = 255
                return mask_full, safe_bbox

            safe_bbox = (fb_x, fb_y, max(bw, fb_w), max(bh, fb_h))
            mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = filled_roi * 255
            return mask_full, safe_bbox

        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        mask_full[by:by+bh, bx:bx+bw] = 255
        return mask_full, cluster_bbox

    def _cluster_heuristic_boxes(self, boxes: List[Tuple[int, int, int, int]], gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Logika penggabungan baris teks lama berbasis ukuran font median dan white connected components.
        """
        if not boxes:
            return []

        h_orig, w_orig = gray.shape
        box_min_dims = [min(b[2], b[3]) for b in boxes]
        median_font_size = float(np.median(box_min_dims)) if box_min_dims else 8.0
        max_dist = max(4, int(median_font_size * 1.35))

        _, white_bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        num_comps, comp_labels = cv2.connectedComponents(white_bin)

        box_comps = []
        for b in boxes:
            bx, by, bw, bh = b
            roi_comps = comp_labels[by:by+bh, bx:bx+bw]
            non_zero = roi_comps[roi_comps > 0]
            cid = int(np.bincount(non_zero).argmax()) if len(non_zero) > 0 else 0
            box_comps.append(cid)

        n = len(boxes)
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if box_comps[i] > 0 and box_comps[j] > 0 and box_comps[i] != box_comps[j]:
                    continue

                b1, b2 = boxes[i], boxes[j]
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
                        first_b = boxes[comp[0]]
                        neigh_b = boxes[neighbor]
                        chain_dx = max(0, max(first_b[0], neigh_b[0]) - min(first_b[0] + first_b[2], neigh_b[0] + neigh_b[2]))
                        chain_dy = max(0, max(first_b[1], neigh_b[1]) - min(first_b[1] + first_b[3], neigh_b[1] + neigh_b[3]))
                        if chain_dx <= max_dist * 4 and chain_dy <= max_dist * 6:
                            visited.add(neighbor)
                            q.append(neighbor)

            xs1 = min([boxes[k][0] for k in comp])
            ys1 = min([boxes[k][1] for k in comp])
            xs2 = max([boxes[k][0] + boxes[k][2] for k in comp])
            ys2 = max([boxes[k][1] + boxes[k][3] for k in comp])
            clusters.append((xs1, ys1, xs2 - xs1, ys2 - ys1))

        return clusters

    def _detect_heuristic(
        self,
        img_rgb: np.ndarray,
        raw_boxes: List[Tuple[int, int, int, int]],
        gray: np.ndarray,
        h_orig: int,
        w_orig: int
    ) -> List[Dict[str, Any]]:
        """
        Pipeline deteksi heuristik penuh (fallback).
        """
        filtered_boxes = []
        for b in raw_boxes:
            bx, by, bw, bh = b
            if by < h_orig * 0.08 and bw > w_orig * 0.45:
                continue
            if (bw * bh) > (w_orig * h_orig * 0.35):
                continue
            filtered_boxes.append(b)

        if not filtered_boxes:
            return []

        clusters = self._cluster_heuristic_boxes(filtered_boxes, gray)
        clusters.sort(key=lambda b: (b[1] // max(30, h_orig // 8), -b[0]))

        bubbles = []
        for text_box in clusters:
            bx, by, bw, bh = text_box
            bubble_mask, safe_bbox = self._compute_bubble_mask(gray, text_box)

            pad = max(2, int(min(bw, bh) * 0.15))
            px = max(0, bx - pad)
            py = max(0, by - pad)
            pw = min(w_orig - px, bw + 2 * pad)
            ph = min(h_orig - py, bh + 2 * pad)
            crop_rgb = img_rgb[py:py+ph, px:px+pw]

            bubbles.append({
                "text_bbox": text_box,
                "bubble_bbox": safe_bbox,
                "safe_bbox": safe_bbox,
                "bubble_mask": bubble_mask,
                "bbox": safe_bbox,
                "crop": Image.fromarray(crop_rgb),
                "area": safe_bbox[2] * safe_bbox[3],
                "source": "heuristic",
                "type": "text_bubble"
            })

        return bubbles

    def detect(
        self,
        image_input,
        include_free_text: bool = True,
        debug: bool = False,
        debug_dir: str = "debug_output"
    ) -> List[Dict[str, Any]]:
        """
        Pipeline Deteksi Terintegrasi:
        1. Ambil kotak balon kata dari model YOLOv8.
        2. Ambil baris teks dari RapidOCR DBNet.
        3. Masukkan tiap baris teks ke balon yang melingkupinya (containment >= 0.60).
        4. Gabungkan baris-baris teks di dalam satu balon menjadi satu text_bbox.
        5. Baris teks di luar balon ditandai type="free_text" dan diproses via merge heuristik.
        6. Hitung bubble_mask dan safe_bbox via flood fill.
        7. Fallback otomatis ke heuristik jika YOLO tidak tersedia / gagal.
        """
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Gagal membaca gambar: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        h_orig, w_orig = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Dapatkan baris teks dari detektor teks (RapidOCR DBNet)
        raw_text_boxes = self._detect_raw_boxes(img_rgb)

        # 2. Cek apakah mode memaksa heuristik
        if self.mode == "heuristic":
            bubbles = self._detect_heuristic(img_rgb, raw_text_boxes, gray, h_orig, w_orig)
            if debug:
                self._save_debug_overlay(img_rgb, bubbles, raw_text_boxes, debug_dir, image_input)
            return bubbles

        # 3. Jalankan YOLOv8 bubble detector
        yolo_detections = []
        try:
            yolo_detections = self._detect_yolo_bubbles(img_rgb)
        except Exception as e:
            print(f"[WARN] Error pada eksekusi YOLOv8: {e}. Mengalihkan ke heuristik.")

        # Fallback jika YOLO gagal atau tidak mendeteksi apa pun padahal ada teks
        if not yolo_detections and self.mode in ["auto", "heuristic"]:
            bubbles = self._detect_heuristic(img_rgb, raw_text_boxes, gray, h_orig, w_orig)
            if debug:
                self._save_debug_overlay(img_rgb, bubbles, raw_text_boxes, debug_dir, image_input)
            return bubbles

        # Pisahkan deteksi balon dan free text dari YOLO
        yolo_bubbles = [d[:4] for d in yolo_detections if d[4] == "text_bubble"]
        # Jika YOLO mendeteksi free_text, simpan kotaknya juga
        yolo_free_boxes = [d[:4] for d in yolo_detections if d[4] != "text_bubble"]

        # 4. Asosiasikan baris teks RapidOCR ke balon YOLO (containment >= 0.60)
        bubble_texts: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(len(yolo_bubbles))]
        unassigned_text_boxes = []

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

            if best_containment >= 0.60 and best_bubble_idx >= 0:
                bubble_texts[best_bubble_idx].append(tb)
            else:
                unassigned_text_boxes.append(tb)

        # 5. Bangun objek balon untuk tiap YOLO bubble
        bubbles = []
        for b_idx, yb in enumerate(yolo_bubbles):
            bx, by, bw, bh = yb
            assigned_texts = bubble_texts[b_idx]

            if assigned_texts:
                # Gabungkan baris teks di dalam balon menjadi satu text_bbox
                tx1 = min(t[0] for t in assigned_texts)
                ty1 = min(t[1] for t in assigned_texts)
                tx2 = max(t[0] + t[2] for t in assigned_texts)
                ty2 = max(t[1] + t[3] for t in assigned_texts)
                text_bbox = (tx1, ty1, max(4, tx2 - tx1), max(4, ty2 - ty1))
            else:
                # Jika tidak ada baris teks terdeteksi di dalam balon, gunakan area tengah balon
                pad_w = int(bw * 0.15)
                pad_h = int(bh * 0.15)
                text_bbox = (bx + pad_w, by + pad_h, max(4, bw - 2 * pad_w), max(4, bh - 2 * pad_h))

            # Hitung bubble mask dan safe_bbox lewat flood fill
            bubble_mask, safe_bbox = self._compute_bubble_mask(gray, yb)

            # Crop teks untuk OCR (gunakan text_bbox dengan padding kecil)
            cx, cy, cw, ch = text_bbox
            crop_pad = max(2, int(min(cw, ch) * 0.12))
            px = max(0, cx - crop_pad)
            py = max(0, cy - crop_pad)
            pw = min(w_orig - px, cw + 2 * crop_pad)
            ph = min(h_orig - py, ch + 2 * crop_pad)
            crop_rgb = img_rgb[py:py+ph, px:px+pw]

            bubbles.append({
                "text_bbox": text_bbox,
                "bubble_bbox": yb,
                "safe_bbox": safe_bbox,
                "bubble_mask": bubble_mask,
                "bbox": yb,
                "crop": Image.fromarray(crop_rgb),
                "area": bw * bh,
                "source": "yolo",
                "type": "text_bubble"
            })

        # 6. Tangani teks di luar balon (free_text: narasi, SFX, caption) jika diaktifkan
        if include_free_text and unassigned_text_boxes:
            free_clusters = self._cluster_heuristic_boxes(unassigned_text_boxes, gray)
            for fc in free_clusters:
                fx, fy, fw, fh = fc
                # Abaikan banner raksasa di tepi atas
                if fy < h_orig * 0.08 and fw > w_orig * 0.45:
                    continue
                if (fw * fh) > (w_orig * h_orig * 0.35):
                    continue

                free_mask, free_safe_bbox = self._compute_bubble_mask(gray, fc)

                crop_pad = max(2, int(min(fw, fh) * 0.12))
                px = max(0, fx - crop_pad)
                py = max(0, fy - crop_pad)
                pw = min(w_orig - px, fw + 2 * crop_pad)
                ph = min(h_orig - py, fh + 2 * crop_pad)
                crop_rgb = img_rgb[py:py+ph, px:px+pw]

                bubbles.append({
                    "text_bbox": fc,
                    "bubble_bbox": free_safe_bbox,
                    "safe_bbox": free_safe_bbox,
                    "bubble_mask": free_mask,
                    "bbox": free_safe_bbox,
                    "crop": Image.fromarray(crop_rgb),
                    "area": fw * fh,
                    "source": "yolo",
                    "type": "free_text"
                })

        # 7. Urutkan urutan baca manga (atas ke bawah, kanan ke kiri)
        bubbles.sort(key=lambda b: (b["bbox"][1] // max(30, h_orig // 8), -b["bbox"][0]))

        # 8. Simpan debug overlay jika diminta
        if debug:
            self._save_debug_overlay(img_rgb, bubbles, raw_text_boxes, debug_dir, image_input)

        return bubbles

    def _save_debug_overlay(
        self,
        img_rgb: np.ndarray,
        bubbles: List[Dict[str, Any]],
        raw_boxes: List[Tuple[int, int, int, int]],
        debug_dir: str,
        image_input
    ):
        os.makedirs(debug_dir, exist_ok=True)
        overlay = img_rgb.copy()

        # Mask hijau transparan
        mask_color = np.zeros_like(img_rgb)
        for b in bubbles:
            if "bubble_mask" in b and b["bubble_mask"] is not None:
                mask_color[b["bubble_mask"] > 0] = [0, 220, 100]

        alpha = 0.28
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0)

        # Gambar raw DBNet boxes (Biru)
        for rb in raw_boxes:
            rx, ry, rw, rh = rb
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (50, 120, 255), 1)

        # Gambar bubble bbox (Hijau terang untuk balon, Oranye untuk free_text) dan text_bbox (Merah)
        for i, b in enumerate(bubbles):
            tx, ty, tw, th = b["text_bbox"]
            cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (255, 60, 60), 1)

            bx, by, bw, bh = b["bubble_bbox"]
            color = (0, 255, 0) if b.get("type") == "text_bubble" else (255, 165, 0)
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(
                overlay,
                f"#{i+1} {b.get('source')}:{b.get('type','b')[:4]}",
                (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        debug_filename = "debug_overlay.png"
        if isinstance(image_input, str):
            debug_filename = f"debug_{os.path.basename(image_input)}"
        debug_path = os.path.join(debug_dir, debug_filename)
        Image.fromarray(overlay).save(debug_path)
        print(f"[DEBUG] Overlay deteksi disimpan ke: {debug_path}")

if __name__ == "__main__":
    detector = BubbleDetector(mode="auto")
    res = detector.detect("evaluation/images/wikipe_page1.jpg", debug=True, debug_dir="debug_output")
    print(f"Total balon terdeteksi: {len(res)}")
    for i, b in enumerate(res):
        print(f"#{i+1} [{b['source']}|{b['type']}]: text={b['text_bbox']}, bubble={b['bubble_bbox']}")
