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
        Untuk strip vertikal panjang (rasio aspek > 2.0, e.g. manhwa/webtoon), gunakan
        tiling vertikal ber-overlap + NMS agar resolusi vertikal tidak terkompresi.
        """
        model = self._get_yolo_model()
        if model is None:
            return []

        names = model.names or {}
        has_text_bubble = any(str(v).lower() == "text_bubble" for v in names.values())
        if not has_text_bubble:
            print(f"[YOLO] Kelas 'text_bubble' tidak ditemukan dalam model.names ({names}). Memperlakukan semua kelas sebagai balon kata.")

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
                    raw_name = str(names.get(cls_id, cls_id))
                    cls_name = raw_name if (has_text_bubble and raw_name.lower() == "text_bubble") else "text_bubble"
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
            raw_name = str(names.get(cls_id, cls_id))
            cls_name = raw_name if (has_text_bubble and raw_name.lower() == "text_bubble") else "text_bubble"
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bw = max(4, x2 - x1)
            bh = max(4, y2 - y1)
            bubble_boxes.append((x1, y1, bw, bh, cls_name, conf))

        return bubble_boxes

    def _grow_safe_bbox_integral(
        self,
        mask: np.ndarray,
        text_bbox: Tuple[int, int, int, int],
        min_density: float = 0.97
    ) -> Tuple[int, int, int, int]:
        """
        Tumbuhkan safe_bbox dari text_bbox menggunakan cv2.integral:
        1. Pastikan area text_bbox diisi 1 agar goresan huruf gelap tidak membuat lubang.
        2. Perbesar ke 4 arah (kiri, kanan, atas, bawah) selama strip baru >= min_density (97%).
        3. Beroperasi dalam O(1) per langkah ekspansi tanpa loop bersarang Python.
        """
        H, W = mask.shape[:2]
        if H == 0 or W == 0:
            return text_bbox

        tx, ty, tw, th = text_bbox
        x1 = max(0, min(W - 1, tx))
        y1 = max(0, min(H - 1, ty))
        x2 = max(0, min(W - 1, tx + tw - 1))
        y2 = max(0, min(H - 1, ty + th - 1))

        if x2 < x1 or y2 < y1:
            return text_bbox

        bin_mask = (mask > 0).astype(np.uint8)
        # Stamp area text_bbox sebagai bagian dari interior balon
        bin_mask[y1:y2+1, x1:x2+1] = 1

        integral = cv2.integral(bin_mask)

        def get_sum(rx1, ry1, rx2, ry2):
            return int(integral[ry2 + 1, rx2 + 1] - integral[ry1, rx2 + 1] - integral[ry2 + 1, rx1] + integral[ry1, rx1])

        expanded = True
        while expanded:
            expanded = False

            # Kiri
            if x1 > 0:
                nx1 = x1 - 1
                strip_sum = get_sum(nx1, y1, nx1, y2)
                strip_len = y2 - y1 + 1
                if strip_sum / strip_len >= min_density:
                    x1 = nx1
                    expanded = True

            # Kanan
            if x2 < W - 1:
                nx2 = x2 + 1
                strip_sum = get_sum(nx2, y1, nx2, y2)
                strip_len = y2 - y1 + 1
                if strip_sum / strip_len >= min_density:
                    x2 = nx2
                    expanded = True

            # Atas
            if y1 > 0:
                ny1 = y1 - 1
                strip_sum = get_sum(x1, ny1, x2, ny1)
                strip_len = x2 - x1 + 1
                if strip_sum / strip_len >= min_density:
                    y1 = ny1
                    expanded = True

            # Bawah
            if y2 < H - 1:
                ny2 = y2 + 1
                strip_sum = get_sum(x1, ny2, x2, ny2)
                strip_len = x2 - x1 + 1
                if strip_sum / strip_len >= min_density:
                    y2 = ny2
                    expanded = True

        return (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))

    def _compute_bubble_mask(
        self,
        gray: np.ndarray,
        text_bbox: Tuple[int, int, int, int],
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        font_size: Optional[float] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Hitung bubble mask berbasis komponen terang terhubung (bright connected component):
        1. ROI dibatasi oleh roi_bbox (dari YOLO) atau ekspansi bertahap dari text_bbox.
        2. Kecerahan interior diukur dari area text_bbox (garansi berada di dalam balon).
        3. Thresholding adaptif untuk mengekstrak interior terang balon.
        4. Morphological opening proporsional font_size untuk memutus ekor tipis dan celah sempit.
        5. Ekstraksi komponen terang yang melingkupi/berpusat pada text_bbox.
        6. Guard kebocoran: deteksi apakah komponen menyentuh batas internal ROI. Jika bocor setelah retry, fallback.
        7. safe_bbox ditumbuhkan dari text_bbox menggunakan cv2.integral (O(1) strip sum).
        """
        h_img, w_img = gray.shape[:2]
        tx, ty, tw, th = text_bbox
        f_size = font_size if font_size is not None else max(8.0, float(min(tw, th) * 0.5))

        def extract_component_in_roi(rx1, ry1, rx2, ry2):
            roi_gray = gray[ry1:ry2, rx1:rx2]
            rh, rw = roi_gray.shape
            if rh < 4 or rw < 4:
                return None, None, None, None

            rel_tx = max(0, tx - rx1)
            rel_ty = max(0, ty - ry1)
            rel_tw = min(rw - rel_tx, tw)
            rel_th = min(rh - rel_ty, th)

            # Ukur kecerahan interior di dalam text_bbox
            text_roi = roi_gray[rel_ty:rel_ty+rel_th, rel_tx:rel_tx+rel_tw]
            if text_roi.size > 0:
                bg_bright = float(np.percentile(text_roi, 85))
            else:
                bg_bright = float(np.percentile(roi_gray, 85))

            thresh = max(130, int(bg_bright - 35))
            bright_mask = (roi_gray >= thresh).astype(np.uint8) * 255

            # Kernel minimal 3x3 agar memutus ekor tipis tanpa merusak sudut kotak narasi atau balon runcing/screaming
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bright_opened = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_opened)

            best_label = 0
            if rel_tw > 0 and rel_th > 0:
                text_labels = labels[rel_ty:rel_ty+rel_th, rel_tx:rel_tx+rel_tw]
                non_zero = text_labels[text_labels > 0]
                if len(non_zero) > 0:
                    counts = np.bincount(non_zero)
                    best_label = int(np.argmax(counts))

            if best_label == 0:
                tcx = rel_tx + rel_tw / 2.0 if rel_tw > 0 else rw / 2.0
                tcy = rel_ty + rel_th / 2.0 if rel_th > 0 else rh / 2.0
                min_dist = float('inf')
                for lbl in range(1, num_labels):
                    cx, cy = centroids[lbl]
                    d = np.hypot(cx - tcx, cy - tcy)
                    if d < min_dist:
                        min_dist = d
                        best_label = lbl

            if best_label > 0:
                bubble_comp = (labels == best_label).astype(np.uint8) * 255
                bubble_comp = cv2.morphologyEx(bubble_comp, cv2.MORPH_DILATE, kernel_open)
                bubble_comp = cv2.bitwise_and(bubble_comp, bright_mask)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                bubble_comp = cv2.morphologyEx(bubble_comp, cv2.MORPH_CLOSE, kernel_close)
                # Isi lubang interior (goresan teks gelap) di dalam balon
                contours, _ = cv2.findContours(bubble_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(largest_cnt)
                    area_cnt = cv2.contourArea(largest_cnt)
                    area_hull = cv2.contourArea(hull)
                    # Jika bentuk mendekati konveks (balon standar / kotak narasi), gunakan convex hull
                    # agar goresan teks yang menempel ke outline tidak mengikis rongga balon
                    if area_hull > 0 and (area_cnt / area_hull) >= 0.70:
                        cv2.drawContours(bubble_comp, [hull], -1, 255, -1)
                    else:
                        cv2.drawContours(bubble_comp, contours, -1, 255, -1)
            else:
                bubble_comp = np.zeros((rh, rw), dtype=np.uint8)
                bubble_comp[rel_ty:rel_ty+rel_th, rel_tx:rel_tx+rel_tw] = 255

            return bubble_comp, (rel_tx, rel_ty, rel_tw, rel_th), rh, rw

        # Tentukan ROI awal
        if roi_bbox is not None:
            bx, by, bw, bh = roi_bbox
            pad_x = max(4, int(bw * 0.05))
            pad_y = max(4, int(bh * 0.05))
            rx1 = max(0, min(bx, tx) - pad_x)
            ry1 = max(0, min(by, ty) - pad_y)
            rx2 = min(w_img, max(bx + bw, tx + tw) + pad_x)
            ry2 = min(h_img, max(by + bh, ty + th) + pad_y)
        else:
            pad_x = max(16, int(f_size * 2.5), int(tw * 0.40))
            pad_y = max(14, int(f_size * 2.0), int(th * 0.35))
            rx1 = max(0, tx - pad_x)
            ry1 = max(0, ty - pad_y)
            rx2 = min(w_img, tx + tw + pad_x)
            ry2 = min(h_img, ty + th + pad_y)

        bubble_roi, rel_tbbox, rh, rw = extract_component_in_roi(rx1, ry1, rx2, ry2)

        if bubble_roi is None:
            mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            mask_full[ty:ty+th, tx:tx+tw] = 255
            return mask_full, text_bbox

        touches_internal_edge = False
        if roi_bbox is None:
            # Guard kebocoran hanya berlaku pada mode heuristik tanpa pembatas YOLO
            touches_internal_edge = (
                (rx1 > 0 and np.any(bubble_roi[:, 0] > 0)) or
                (rx2 < w_img and np.any(bubble_roi[:, -1] > 0)) or
                (ry1 > 0 and np.any(bubble_roi[0, :] > 0)) or
                (ry2 < h_img and np.any(bubble_roi[-1, :] > 0))
            )

        if touches_internal_edge:
            # Percobaan 2: Perbesar ROI bertahap
            extra_pad_x = int(f_size * 2.0)
            extra_pad_y = int(f_size * 2.0)
            erx1 = max(0, rx1 - extra_pad_x)
            ery1 = max(0, ry1 - extra_pad_y)
            erx2 = min(w_img, rx2 + extra_pad_x)
            ery2 = min(h_img, ry2 + extra_pad_y)

            exp_bubble_roi, exp_rel_tbbox, erh, erw = extract_component_in_roi(erx1, ery1, erx2, ery2)
            if exp_bubble_roi is not None:
                still_touches = (
                    (erx1 > 0 and np.any(exp_bubble_roi[:, 0] > 0)) or
                    (erx2 < w_img and np.any(exp_bubble_roi[:, -1] > 0)) or
                    (ery1 > 0 and np.any(exp_bubble_roi[0, :] > 0)) or
                    (ery2 < h_img and np.any(exp_bubble_roi[-1, :] > 0))
                )
                if not still_touches:
                    # Berhasil terisolasi di dalam ROI yang diperbesar
                    rx1, ry1, rx2, ry2 = erx1, ery1, erx2, ery2
                    bubble_roi, rel_tbbox, rh, rw = exp_bubble_roi, exp_rel_tbbox, erh, erw
                else:
                    # Tetap menyentuh: fallback ke kotak teks + padding terisolasi
                    pad_fallback = max(2, int(f_size * 0.40))
                    fb_x = max(0, tx - pad_fallback)
                    fb_y = max(0, ty - pad_fallback)
                    fb_w = min(w_img - fb_x, tw + 2 * pad_fallback)
                    fb_h = min(h_img - fb_y, th + 2 * pad_fallback)
                    safe_bbox = (fb_x, fb_y, fb_w, fb_h)

                    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
                    mask_full[fb_y:fb_y+fb_h, fb_x:fb_x+fb_w] = 255
                    return mask_full, safe_bbox

        # Hitung safe_bbox dengan menumbuhkan dari text_bbox menggunakan cv2.integral
        srx, sry, srw, srh = self._grow_safe_bbox_integral(bubble_roi, rel_tbbox, min_density=0.97)
        safe_bbox = (int(rx1 + srx), int(ry1 + sry), int(srw), int(srh))

        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        mask_full[ry1:ry2, rx1:rx2] = bubble_roi

        return mask_full, safe_bbox

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

    @staticmethod
    def _prepare_ocr_crop(
        img_rgb: np.ndarray,
        crop_bbox: Tuple[int, int, int, int],
        bubble_mask: Optional[np.ndarray] = None,
        font_size: float = 16.0,
        padding: int = 10,
        target_font_height: float = 28.0
    ) -> Image.Image:
        """
        Pra-proses crop balon kata sebelum dikirim ke engine OCR:
        1. Ambil potongan gambar sesuai crop_bbox (safe_bbox).
        2. Jika bubble_mask tersedia: erode mask 2-3px lalu jadikan putih (255)
           semua piksel di luar mask untuk membuang outline dan latar artwork.
        3. Tambahkan padding putih (8-12 px) di sekeliling crop.
        4. Skalakan agar tinggi huruf optimal sekitar 24-32 px (default target 28 px).
        """
        cx, cy, cw, ch = crop_bbox
        h_orig, w_orig = img_rgb.shape[:2]
        px, py = max(0, cx), max(0, cy)
        pw, ph = min(w_orig - px, cw), min(h_orig - py, ch)

        if pw <= 0 or ph <= 0:
            return Image.new("RGB", (32, 32), color=(255, 255, 255))

        crop = img_rgb[py:py+ph, px:px+pw].copy()

        # 1. Bersihkan piksel di luar balon kata menjadi putih murni
        if bubble_mask is not None and bubble_mask.shape[:2] == (h_orig, w_orig):
            mask_slice = bubble_mask[py:py+ph, px:px+pw]
            kernel_size = max(3, int(font_size * 0.15)) | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            eroded_mask = cv2.erode(mask_slice, kernel, iterations=1)
            crop[eroded_mask == 0] = [255, 255, 255]

        # 2. Tambahkan padding putih 8-12 px di sekeliling crop
        crop = cv2.copyMakeBorder(
            crop,
            padding, padding, padding, padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

        # 3. Skalakan agar tinggi huruf sekitar 24-32 px (bukan upscale mentah ke 736 px)
        current_font = max(8.0, float(font_size))
        scale = target_font_height / current_font
        scale = max(0.6, min(2.5, scale))

        if abs(scale - 1.0) > 0.08:
            new_w = max(16, int(round(crop.shape[1] * scale)))
            new_h = max(16, int(round(crop.shape[0] * scale)))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        return Image.fromarray(crop)

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

        box_min_dims = [min(b[2], b[3]) for b in filtered_boxes]
        median_font_size = float(np.median(box_min_dims)) if box_min_dims else 14.0

        bubbles = []
        for text_box in clusters:
            bx, by, bw, bh = text_box
            bubble_mask, safe_bbox = self._compute_bubble_mask(gray, text_box, roi_bbox=None, font_size=median_font_size)

            # Crop seluruh rongga dalam balon kata (safe_bbox) untuk OCR dengan pra-proses
            crop_img = self._prepare_ocr_crop(
                img_rgb=img_rgb,
                crop_bbox=safe_bbox,
                bubble_mask=bubble_mask,
                font_size=median_font_size,
                padding=10,
                target_font_height=28.0
            )

            bubbles.append({
                "text_bbox": text_box,
                "bubble_bbox": safe_bbox,
                "safe_bbox": safe_bbox,
                "bubble_mask": bubble_mask,
                "bbox": safe_bbox,
                "crop": crop_img,
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
        box_min_dims = [min(b[2], b[3]) for b in raw_text_boxes]
        median_font_size = float(np.median(box_min_dims)) if box_min_dims else 14.0

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
            assigned_texts = list(bubble_texts[b_idx])

            # Deteksi teks presisi pada potongan (crop) balon HANYA jika DBNet skala penuh belum menemukan teks di balon ini
            if not assigned_texts:
                bubble_crop = img_rgb[by:by+bh, bx:bx+bw]
                crop_dt_boxes, _ = self.engine.text_det(bubble_crop)
                if crop_dt_boxes is not None and len(crop_dt_boxes) > 0:
                    for b in crop_dt_boxes:
                        xs = [pt[0] for pt in b]
                        ys = [pt[1] for pt in b]
                        cbx = int(max(0, min(xs)))
                        cby = int(max(0, min(ys)))
                        cbw = int(max(xs) - min(xs))
                        cbh = int(max(ys) - min(ys))
                        cand_box = (bx + cbx, by + cby, cbw, cbh)
                        # Pastikan kandidat tidak bertumpang tindih dengan teks balon lain
                        is_foreign = False
                        for other_idx, other_boxes in enumerate(bubble_texts):
                            if other_idx == b_idx:
                                continue
                            for otb in other_boxes:
                                ox, oy, ow, oh = otb
                                if max(cand_box[0], ox) < min(cand_box[0] + cand_box[2], ox + ow) and \
                                   max(cand_box[1], oy) < min(cand_box[1] + cand_box[3], oy + oh):
                                    is_foreign = True
                                    break
                            if is_foreign:
                                break
                        if not is_foreign and cbw >= 4 and cbh >= 4:
                            assigned_texts.append(cand_box)

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

            # Hitung bubble mask dan safe_bbox berbasis bright connected components
            bubble_mask, safe_bbox = self._compute_bubble_mask(
                gray=gray,
                text_bbox=text_bbox,
                roi_bbox=yb,
                font_size=median_font_size
            )

            # Mask teks milik balon tetangga/asing pada gambar crop OCR agar tidak terbaca campur aduk
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
                        px2 = min(w_orig, ox + ow + 4)
                        py2 = min(h_orig, oy + oh + 4)
                        cleaned_rgb[py1:py2, px1:px2] = 255

            # Crop seluruh isi rongga balon kata (safe_bbox) untuk OCR dengan pra-proses
            crop_img = self._prepare_ocr_crop(
                img_rgb=cleaned_rgb,
                crop_bbox=safe_bbox,
                bubble_mask=bubble_mask,
                font_size=median_font_size,
                padding=10,
                target_font_height=28.0
            )

            bubbles.append({
                "text_bbox": text_bbox,
                "bubble_bbox": yb,
                "safe_bbox": safe_bbox,
                "bubble_mask": bubble_mask,
                "bbox": safe_bbox,
                "crop": crop_img,
                "area": safe_bbox[2] * safe_bbox[3],
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

                free_mask, free_safe_bbox = self._compute_bubble_mask(
                    gray=gray,
                    text_bbox=fc,
                    roi_bbox=None,
                    font_size=median_font_size
                )

                # Periksa berapa kotak teks penyusun cluster ini
                boxes_in_cluster = [
                    b for b in unassigned_text_boxes
                    if b[0] >= fx - 12 and b[1] >= fy - 12 and (b[0] + b[2]) <= (fx + fw + 12) and (b[1] + b[3]) <= (fy + fh + 12)
                ]

                # Jika cluster memiliki >= 2 baris teks bertumpuk atau cukup lebar untuk sebuah kalimat,
                # ini adalah balon kata dialog / kotak narasi (cth: balon berbayang screentone abu-abu)
                # Klasifikasikan sebagai 'text_bubble' agar selalu diterjemahkan & di-render!
                is_stacked_dialogue = len(boxes_in_cluster) >= 2 or (fw >= 65 and fh >= 20)
                b_type = "text_bubble" if is_stacked_dialogue else "free_text"
                b_source = "dbnet_cluster" if is_stacked_dialogue else "yolo"

                # Pastikan safe_bbox melingkupi teks cluster dengan margin yang nyaman
                sx = min(free_safe_bbox[0], max(0, fx - 10))
                sy = min(free_safe_bbox[1], max(0, fy - 10))
                sx2 = max(free_safe_bbox[0] + free_safe_bbox[2], min(w_orig, fx + fw + 10))
                sy2 = max(free_safe_bbox[1] + free_safe_bbox[3], min(h_orig, fy + fh + 10))
                free_safe_bbox = (sx, sy, sx2 - sx, sy2 - sy)

                crop_pad = max(2, int(min(fw, fh) * 0.12))
                crop_box = (max(0, fx - crop_pad), max(0, fy - crop_pad), fw + 2 * crop_pad, fh + 2 * crop_pad)
                crop_img = self._prepare_ocr_crop(
                    img_rgb=img_rgb,
                    crop_bbox=crop_box,
                    bubble_mask=free_mask,
                    font_size=median_font_size,
                    padding=10,
                    target_font_height=28.0
                )

                bubbles.append({
                    "text_bbox": fc,
                    "bubble_bbox": free_safe_bbox,
                    "safe_bbox": free_safe_bbox,
                    "bubble_mask": free_mask,
                    "bbox": free_safe_bbox,
                    "crop": crop_img,
                    "area": free_safe_bbox[2] * free_safe_bbox[3],
                    "source": b_source,
                    "type": b_type
                })

        # 7. Selesaikan tabrakan antar-balon yang menempel / overlapping (connected bubbles)
        self._resolve_connected_bubbles(bubbles)

        # 8. Urutkan urutan baca manga (atas ke bawah, kanan ke kiri)
        bubbles.sort(key=lambda b: (b["bbox"][1] // max(30, h_orig // 8), -b["bbox"][0]))

        # 9. Simpan debug overlay jika diminta
        if debug:
            self._save_debug_overlay(img_rgb, bubbles, raw_text_boxes, debug_dir, image_input)

        return bubbles

    def _resolve_connected_bubbles(self, bubbles: List[Dict[str, Any]]):
        """
        Selesaikan tabrakan antar-balon kata yang berdekatan atau menempel (connected bubbles):
        Jika dua balon safe_bbox-nya bertabrakan/berpotongan, pisahkan safe_bbox dan mask
        pada garis batas tengah antar teks, sehingga teks terjemahan kedua balon tidak pernah
        saling menimpa (collide).
        """
        n = len(bubbles)
        for i in range(n):
            for j in range(i + 1, n):
                b1 = bubbles[i]
                b2 = bubbles[j]
                if b1.get("type") != "text_bubble" or b2.get("type") != "text_bubble":
                    continue
                s1 = b1.get("safe_bbox")
                s2 = b2.get("safe_bbox")
                if not s1 or not s2:
                    continue

                xi1 = max(s1[0], s2[0])
                yi1 = max(s1[1], s2[1])
                xi2 = min(s1[0] + s1[2], s2[0] + s2[2])
                yi2 = min(s1[1] + s1[3], s2[1] + s2[3])

                if xi2 > xi1 and yi2 > yi1:
                    t1 = b1["text_bbox"]
                    t2 = b2["text_bbox"]
                    c1x, c1y = t1[0] + t1[2] / 2.0, t1[1] + t1[3] / 2.0
                    c2x, c2y = t2[0] + t2[2] / 2.0, t2[1] + t2[3] / 2.0

                    dx = abs(c1x - c2x)
                    dy = abs(c1y - c2y)

                    if dx >= dy:
                        # Terpisah secara dominan horizontal
                        if c1x < c2x:
                            left_b, right_b = b1, b2
                        else:
                            left_b, right_b = b2, b1

                        lt = left_b["text_bbox"]
                        rt = right_b["text_bbox"]
                        split_x = int(((lt[0] + lt[2]) + rt[0]) / 2.0)

                        ls = left_b["safe_bbox"]
                        new_lw = max(lt[2] + 4, split_x - ls[0])
                        left_b["safe_bbox"] = (ls[0], ls[1], new_lw, ls[3])
                        left_b["bbox"] = left_b["safe_bbox"]
                        if left_b.get("bubble_mask") is not None:
                            left_b["bubble_mask"][:, split_x:] = 0

                        rs = right_b["safe_bbox"]
                        new_rx = max(rs[0], split_x)
                        new_rw = max(rt[2] + 4, (rs[0] + rs[2]) - new_rx)
                        right_b["safe_bbox"] = (new_rx, rs[1], new_rw, rs[3])
                        right_b["bbox"] = right_b["safe_bbox"]
                        if right_b.get("bubble_mask") is not None:
                            right_b["bubble_mask"][:, :split_x] = 0
                    else:
                        # Terpisah secara dominan vertikal
                        if c1y < c2y:
                            top_b, bottom_b = b1, b2
                        else:
                            top_b, bottom_b = b2, b1

                        tt = top_b["text_bbox"]
                        bt = bottom_b["text_bbox"]
                        split_y = int(((tt[1] + tt[3]) + bt[1]) / 2.0)

                        ts = top_b["safe_bbox"]
                        new_th = max(tt[3] + 4, split_y - ts[1])
                        top_b["safe_bbox"] = (ts[0], ts[1], ts[2], new_th)
                        top_b["bbox"] = top_b["safe_bbox"]
                        if top_b.get("bubble_mask") is not None:
                            top_b["bubble_mask"][split_y:, :] = 0

                        bs = bottom_b["safe_bbox"]
                        new_by = max(bs[1], split_y)
                        new_bh = max(bt[3] + 4, (bs[1] + bs[3]) - new_by)
                        bottom_b["safe_bbox"] = (bs[0], new_by, bs[2], new_bh)
                        bottom_b["bbox"] = bottom_b["safe_bbox"]
                        if bottom_b.get("bubble_mask") is not None:
                            bottom_b["bubble_mask"][:split_y, :] = 0

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
