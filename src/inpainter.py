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
                med_val = float(np.median(interior_gray))
                mean_val = float(np.mean(interior_gray))
                is_dark_bubble = (med_val < 110 or mean_val < 110)

                if is_dark_bubble:
                    # Balon gelap / hitam: latar belakang gelap dan teks terang/putih
                    bg_val = int(np.percentile(interior_gray, 15))
                    light_thresh = max(110, int(bg_val + 30))
                    text_pixels = (gray > light_thresh) & interior_idx
                    text_dilated = cv2.dilate(text_pixels.astype(np.uint8), k_dilate, iterations=2)
                    text_to_erase = (text_dilated > 0) & interior_idx
                    # Inpaint teks terang pada latar gelap untuk menjaga gradasi/tekstur balon hitam
                    y_indices, x_indices = np.where(interior_idx)
                    by1, by2 = int(np.min(y_indices)), int(np.max(y_indices) + 1)
                    bx1, bx2 = int(np.min(x_indices)), int(np.max(x_indices) + 1)
                    roi_col = clean_img[by1:by2, bx1:bx2]
                    roi_erase = text_to_erase[by1:by2, bx1:bx2].astype(np.uint8)
                    inpainted = cv2.inpaint(roi_col, roi_erase, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                    clean_img[by1:by2, bx1:bx2][roi_erase > 0] = inpainted[roi_erase > 0]
                else:
                    # Balon terang / putih / screentone
                    bg_val = int(np.percentile(interior_gray, 92))
                    fill_color = [bg_val, bg_val, bg_val]

                    # Ukur uniformitas latar belakang (apakah putih polos atau memiliki screentone/tekstur)
                    bg_sample = interior_gray[interior_gray >= max(0, bg_val - 25)]
                    bg_std = float(np.std(bg_sample)) if len(bg_sample) > 0 else 0.0

                    # Deteksi piksel gelap teks di dalam balon
                    dark_thresh = min(225, bg_val - 15) if bg_val >= 170 else max(20, bg_val - 25)
                    dark_pixels = (gray < dark_thresh) & interior_idx

                    # Dilasi 2px untuk menangkap fringe anti-aliasing tepi goresan
                    dark_dilated = cv2.dilate(dark_pixels.astype(np.uint8), k_dilate, iterations=2)
                    dark_to_erase = (dark_dilated > 0) & interior_idx

                    # Isi solid fill HANYA jika latar belakang putih polos seragam (bg_val >= 235 dan std <= 8.0)
                    # Jika berbayang, berpola, atau bertekstur screentone, gunakan cv2.inpaint agar tekstur tetap utuh!
                    if bg_val >= 235 and bg_std <= 8.0:
                        clean_img[dark_to_erase] = fill_color
                    else:
                        y_indices, x_indices = np.where(interior_idx)
                        by1, by2 = int(np.min(y_indices)), int(np.max(y_indices) + 1)
                        bx1, bx2 = int(np.min(x_indices)), int(np.max(x_indices) + 1)
                        roi_col = clean_img[by1:by2, bx1:bx2]
                        roi_erase = dark_to_erase[by1:by2, bx1:bx2].astype(np.uint8)
                        inpainted = cv2.inpaint(roi_col, roi_erase, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                        clean_img[by1:by2, bx1:bx2][roi_erase > 0] = inpainted[roi_erase > 0]

                # Garansi kebersihan interior safe_bbox untuk kotak narasi / balon polos
                if safe_bboxes is not None and i < len(safe_bboxes) and safe_bboxes[i] is not None:
                    sx, sy, sw, sh = safe_bboxes[i]
                    margin = 2
                    sx1, sy1 = max(0, sx + margin), max(0, sy + margin)
                    sx2, sy2 = min(w_img, sx + sw - margin), min(h_img, sy + sh - margin)
                    if sx2 > sx1 and sy2 > sy1 and not is_dark_bubble and bg_val >= 235 and bg_std <= 8.0:
                        safe_dark = (gray[sy1:sy2, sx1:sx2] < dark_thresh)
                        clean_img[sy1:sy2, sx1:sx2][safe_dark] = fill_color

                # Garansi kebersihan text_bbox asli agar goresan teks lama tidak pernah tertinggal pada balon polos
                tx1, ty1 = max(0, x - 2), max(0, y - 2)
                tx2, ty2 = min(w_img, x + w + 2), min(h_img, y + h + 2)
                if tx2 > tx1 and ty2 > ty1 and not is_dark_bubble and bg_val >= 235 and bg_std <= 8.0:
                    text_dark = (gray[ty1:ty2, tx1:tx2] < dark_thresh)
                    clean_img[ty1:ty2, tx1:tx2][text_dark] = fill_color

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
            bg_sample = roi_gray[roi_gray >= max(0, bg_val - 25)]
            bg_std = float(np.std(bg_sample)) if len(bg_sample) > 0 else 0.0

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

                if bg_val >= 235 and bg_std <= 8.0:
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = fill_color
                else:
                    inpainted = cv2.inpaint(roi_color, mask_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = inpainted[mask_dilated > 0]

        return Image.fromarray(clean_img)

if __name__ == "__main__":
    print("MangaInpainter stroke-level siap digunakan.")
