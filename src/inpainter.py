import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

class MangaInpainter:
    def __init__(self):
        """
        Inpainter presisi goresan tinta (Stroke-Level Inpainting):
        - HANYA menghapus goresan tinta teks di dalam area teks yang terdeteksi
        - TIDAK PERNAH membuat kotak putih besar
        - TIDAK PERNAH menyentuh garis gambar karakter, border balon, atau artwork
        """
        pass

    def clean_bubbles(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        masks: Optional[List[np.ndarray]] = None,
        safe_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Image.Image:
        """
        Menghapus goresan teks dialog asli dari balon kata:
        1. Jika bubble_mask tersedia: erosi mask setebal outline (3px), cari seluruh piksel gelap
           di dalam rongga balon, dilasi 2-3px untuk membersihkan fringe anti-aliasing, dan hapus/inpaint
           seluruh goresan tanpa pernah menyentuh outline balon.
        2. Jika safe_bboxes tersedia: garansi kebersihan seluruh interior safe_bbox.
        3. Jika bubble_mask tidak tersedia (fallback): gunakan stroke-level connected components
           pada text_bbox.
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clean_img = img_np.copy()
        h_img, w_img = gray.shape

        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for i, (x, y, w, h) in enumerate(bboxes):
            mask_full = masks[i] if (masks is not None and i < len(masks) and masks[i] is not None) else None

            # Mode 1: Hapus seluruh piksel gelap di dalam bubble_mask yang di-erode setebal outline
            if mask_full is not None and np.any(mask_full > 0):
                eroded_mask = cv2.erode((mask_full > 0).astype(np.uint8), k_erode, iterations=1)
                interior_idx = eroded_mask > 0
                if not np.any(interior_idx):
                    continue

                interior_gray = gray[interior_idx]
                bg_val = int(np.percentile(interior_gray, 92))
                fill_color = [bg_val, bg_val, bg_val]

                # Deteksi piksel gelap teks di dalam balon (tanpa cap 190 yang meninggalkan fringe anti-aliasing)
                dark_thresh = min(225, bg_val - 15) if bg_val >= 170 else max(20, bg_val - 25)
                dark_pixels = (gray < dark_thresh) & interior_idx

                # Dilasi 2px untuk menangkap fringe anti-aliasing tepi goresan
                dark_dilated = cv2.dilate(dark_pixels.astype(np.uint8), k_dilate, iterations=2)
                dark_to_erase = (dark_dilated > 0) & interior_idx

                # Isi solid fill jika warna seragam (khas manga) atau inpaint jika berpola
                if bg_val >= 170:
                    clean_img[dark_to_erase] = fill_color
                else:
                    y_indices, x_indices = np.where(interior_idx)
                    by1, by2 = int(np.min(y_indices)), int(np.max(y_indices) + 1)
                    bx1, bx2 = int(np.min(x_indices)), int(np.max(x_indices) + 1)
                    roi_col = clean_img[by1:by2, bx1:bx2]
                    roi_erase = dark_to_erase[by1:by2, bx1:bx2].astype(np.uint8)
                    inpainted = cv2.inpaint(roi_col, roi_erase, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                    clean_img[by1:by2, bx1:bx2][roi_erase > 0] = inpainted[roi_erase > 0]

                # Garansi kebersihan interior safe_bbox untuk kotak narasi / balon
                if safe_bboxes is not None and i < len(safe_bboxes) and safe_bboxes[i] is not None:
                    sx, sy, sw, sh = safe_bboxes[i]
                    margin = 2
                    sx1, sy1 = max(0, sx + margin), max(0, sy + margin)
                    sx2, sy2 = min(w_img, sx + sw - margin), min(h_img, sy + sh - margin)
                    if sx2 > sx1 and sy2 > sy1 and bg_val >= 170:
                        safe_dark = (gray[sy1:sy2, sx1:sx2] < dark_thresh)
                        clean_img[sy1:sy2, sx1:sx2][safe_dark] = fill_color

                continue

            # Mode 2 (Fallback): Stroke-level inpainting berbasis text_bbox
            pad = max(2, int(min(w, h) * 0.08))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            if x2 <= x1 or y2 <= y1:
                continue

            roi_gray = gray[y1:y2, x1:x2]
            roi_color = clean_img[y1:y2, x1:x2]
            rh, rw = roi_gray.shape

            bg_val = int(np.percentile(roi_gray, 90))

            if bg_val < 110:
                thresh_val = 165
                raw_mask = (roi_gray > thresh_val).astype(np.uint8)
                fill_color = [bg_val, bg_val, bg_val]
            else:
                thresh_val = min(180, int(bg_val - 30))
                raw_mask = (roi_gray < thresh_val).astype(np.uint8)
                fill_color = [bg_val, bg_val, bg_val]

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
            clean_text_mask = np.zeros_like(raw_mask)

            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                cx = stats[lbl, cv2.CC_STAT_LEFT]
                cy = stats[lbl, cv2.CC_STAT_TOP]
                cw = stats[lbl, cv2.CC_STAT_WIDTH]
                ch = stats[lbl, cv2.CC_STAT_HEIGHT]

                touches_border = (cx == 0 or cy == 0 or (cx + cw) >= rw or (cy + ch) >= rh)
                if touches_border and (cw > rw * 0.35 or ch > rh * 0.35 or area > rw * rh * 0.20):
                    continue
                if area > (rw * rh * 0.40):
                    continue

                clean_text_mask[labels == lbl] = 255

            mask_pixels = np.sum(clean_text_mask > 0)
            if 0 < mask_pixels < (clean_text_mask.size * 0.75):
                k_size = 2 if min(w_img, h_img) < 1000 else 3
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_dilated = cv2.dilate(clean_text_mask, kernel, iterations=1)

                if bg_val >= 180:
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = fill_color
                else:
                    inpainted = cv2.inpaint(roi_color, mask_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = inpainted[mask_dilated > 0]

        return Image.fromarray(clean_img)

if __name__ == "__main__":
    print("MangaInpainter stroke-level siap digunakan.")
