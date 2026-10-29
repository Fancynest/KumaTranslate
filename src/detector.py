import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Any
from rapidocr_onnxruntime import RapidOCR

class BubbleDetector:
    def __init__(self):
        """
        Detector Balon Kata & Teks Komik Cerdas:
        - Normalisasi resolusi dinamis (optimal untuk gambar kecil 250px maupun scan 4K)
        - Deteksi teks berbasis Deep Learning (DBNet)
        - Penggabungan cerdas multi-arah (mendukung teks vertikal Jepang & horizontal)
        - Filter banner judul chapter & credit agar gambar artwork manga tidak terhapus
        - Penentuan batas balon presisi (Safe Bubble Bounds) yang terhenti di garis outline hitam
        """
        self.engine = RapidOCR()

    def _should_merge(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], max_dist: int) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

        # 1. Dekat di kedua sumbu
        if dx <= max_dist and dy <= max_dist:
            return True

        # 2. Kolom vertikal bertetangga (Komik Jepang / Manga)
        v_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        if v_overlap > 0 and dx <= int(max_dist * 1.5):
            return True

        # 3. Baris horizontal bertumpuk (Komik Barat / Manhua)
        h_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        if h_overlap > 0 and dy <= int(max_dist * 1.5):
            return True

        return False

    def _merge_boxes(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        return (x, y, w, h)

    def _find_safe_bubble_bounds(
        self,
        gray: np.ndarray,
        text_bbox: Tuple[int, int, int, int],
        w_img: int,
        h_img: int
    ) -> Tuple[int, int, int, int]:
        """
        Mencari batas fisik balon kata oval/putih dengan mendeteksi garis outline hitam balon.
        Berhenti tepat di garis outline (<165) sehingga teks tidak pernah melewati garis balon.
        """
        bx, by, bw, bh = text_bbox
        cx = bx + bw // 2
        cy = by + bh // 2

        max_dx = max(8, int(bw * 1.3))
        max_dy = max(6, int(bh * 0.35))

        # Scan ke Kiri (berhenti jika bertemu garis hitam outline < 165)
        left = bx
        while left > max(0, bx - max_dx) and gray[cy, left - 1] > 165:
            left -= 1

        # Scan ke Kanan
        right = bx + bw
        while right < min(w_img, bx + bw + max_dx) and gray[cy, right] > 165:
            right += 1

        # Scan ke Atas
        top = by
        while top > max(0, by - max_dy) and gray[top - 1, cx] > 165:
            top -= 1

        # Scan ke Bawah
        bottom = by + bh
        while bottom < min(h_img, by + bh + max_dy) and gray[bottom, cx] > 165:
            bottom += 1

        return (left, top, max(bw, right - left), max(bh, bottom - top))

    def detect(self, image_input) -> List[Dict[str, Any]]:
        """
        Mendeteksi seluruh balon dialog pada halaman manga tanpa merusak artwork judul/karakter.
        """
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        h_orig, w_orig = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # 1. Normalisasi Skala untuk Deteksi DBNet
        cur_min = min(w_orig, h_orig)
        cur_max = max(w_orig, h_orig)

        if cur_min < 700:
            scale = 750.0 / cur_min
        elif cur_max > 1600:
            scale = 1400.0 / cur_max
        else:
            scale = 1.0

        sw, sh = int(w_orig * scale), int(h_orig * scale)
        if scale != 1.0:
            interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
            det_img = cv2.resize(img_rgb, (sw, sh), interpolation=interp)
        else:
            det_img = img_rgb

        dt_boxes, _ = self.engine.text_det(det_img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        # 2. Kembalikan koordinat ke ukuran asli & filter cerdas
        raw_boxes = []
        for b in dt_boxes:
            xs = [pt[0] / scale for pt in b]
            ys = [pt[1] / scale for pt in b]
            bx = int(max(0, min(xs)))
            by = int(max(0, min(ys)))
            bw = int(min(w_orig - bx, max(xs) - min(xs)))
            bh = int(min(h_orig - by, max(ys) - min(ys)))

            # Abaikan noise titik kecil
            if bw < 4 or bh < 4:
                continue

            # Abaikan banner judul chapter / credit pengarang di bagian header atas
            if by < h_orig * 0.22 and (bw > w_orig * 0.24 or bh > h_orig * 0.18):
                continue

            # Abaikan kotak raksasa (>30% dari seluruh halaman)
            if (bw * bh) > (w_orig * h_orig * 0.30):
                continue

            raw_boxes.append((bx, by, bw, bh))

        if not raw_boxes:
            return []

        # 3. Penggabungan baris teks menjadi 1 balon kata
        max_dist = max(8, int(min(w_orig, h_orig) * 0.025))
        visited = set()
        merged = []

        for i, box in enumerate(raw_boxes):
            if i in visited:
                continue
            curr = box
            visited.add(i)

            changed = True
            while changed:
                changed = False
                for j, other in enumerate(raw_boxes):
                    if j not in visited and self._should_merge(curr, other, max_dist):
                        curr = self._merge_boxes(curr, other)
                        visited.add(j)
                        changed = True

            merged.append(curr)

        # 4. Urutkan berdasarkan urutan baca manga (atas ke bawah, kanan ke kiri)
        merged.sort(key=lambda b: (b[1] // max(30, h_orig // 8), -b[0]))

        # 5. Bangun objek balon lengkap dengan safe bubble_bounds & crop OCR
        bubbles = []
        for bbox in merged:
            bx, by, bw, bh = bbox
            pad = max(2, int(min(bw, bh) * 0.15))
            px = max(0, bx - pad)
            py = max(0, by - pad)
            pw = min(w_orig - px, bw + 2 * pad)
            ph = min(h_orig - py, bh + 2 * pad)

            crop_rgb = img_rgb[py:py+ph, px:px+pw]
            safe_bound = self._find_safe_bubble_bounds(gray, bbox, w_orig, h_orig)

            bubbles.append({
                "text_bbox": bbox,
                "bubble_bbox": safe_bound,
                "bbox": bbox,
                "crop": Image.fromarray(crop_rgb),
                "area": pw * ph
            })

        return bubbles

if __name__ == "__main__":
    detector = BubbleDetector()
    res = detector.detect("temp/5a2a0a45/extracted/page_001.jpg")
    print(f"Total dialog terdeteksi: {len(res)}")
    for i, b in enumerate(res):
        print(f"Balon #{i+1}: text_box={b['text_bbox']} -> safe_bubble_bounds={b['bubble_bbox']}")
