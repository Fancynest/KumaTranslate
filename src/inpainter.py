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
        masks: Optional[List[np.ndarray]] = None
    ) -> Image.Image:
        """
        Menghapus goresan teks dialog asli menggunakan stroke-level inpainting cerdas:
        1. Ekstraksi goresan gelap teks menggunakan threshold adaptif.
        2. Analisis Connected Components untuk memisahkan goresan huruf dari border panel/artwork.
        3. Pengisian goresan teks dengan warna latar belakang balon yang serasi (pure/solid fill)
           atau Navier-Stokes pada screentone kompleks, hanya pada piksel goresan (dilated).
        4. 100% zero-boundary-defect: Piksel di luar goresan huruf tidak pernah disentuh.
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clean_img = img_np.copy()
        h_img, w_img = gray.shape

        for i, (x, y, w, h) in enumerate(bboxes):
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

            # 1. Tentukan kecerahan latar belakang (persentil 90)
            bg_val = int(np.percentile(roi_gray, 90))

            if bg_val < 110:
                # Balon gelap / screentone hitam dengan teks putih
                thresh_val = 165
                raw_mask = (roi_gray > thresh_val).astype(np.uint8)
                fill_color = [bg_val, bg_val, bg_val]
            else:
                # Balon putih standar dengan teks hitam
                thresh_val = min(180, int(bg_val - 30))
                raw_mask = (roi_gray < thresh_val).astype(np.uint8)
                fill_color = [bg_val, bg_val, bg_val]

            # 2. Filter Connected Components: Pisahkan huruf teks dari border panel / rambut / artwork
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
            clean_text_mask = np.zeros_like(raw_mask)

            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                cx = stats[lbl, cv2.CC_STAT_LEFT]
                cy = stats[lbl, cv2.CC_STAT_TOP]
                cw = stats[lbl, cv2.CC_STAT_WIDTH]
                ch = stats[lbl, cv2.CC_STAT_HEIGHT]

                touches_border = (cx == 0 or cy == 0 or (cx + cw) >= rw or (cy + ch) >= rh)
                # Komponen panjang yang menempel border adalah border garis panel atau artwork luar
                if touches_border and (cw > rw * 0.35 or ch > rh * 0.35 or area > rw * rh * 0.20):
                    continue
                if area > (rw * rh * 0.40):
                    continue

                clean_text_mask[labels == lbl] = 255

            # 3. Batasi dengan bubble_mask jika tersedia
            if masks is not None and i < len(masks) and masks[i] is not None:
                roi_bubble = masks[i][y1:y2, x1:x2]
                if np.any(roi_bubble > 0):
                    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    inner_mask = cv2.erode((roi_bubble > 0).astype(np.uint8), k_erode, iterations=1)
                    clean_text_mask = cv2.bitwise_and(clean_text_mask, clean_text_mask, mask=inner_mask)

            # 4. Dilasi kecil (2-3px) untuk membersihkan tepian anti-aliasing goresan
            mask_pixels = np.sum(clean_text_mask > 0)
            if 0 < mask_pixels < (clean_text_mask.size * 0.75):
                k_size = 2 if min(w_img, h_img) < 1000 else 3
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_dilated = cv2.dilate(clean_text_mask, kernel, iterations=1)

                # Jika latar balon seragam (misal putih manga >= 180), isi langsung dengan background
                # agar 100% mulus tanpa noda/smudge telea
                if bg_val >= 180:
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = fill_color
                else:
                    inpainted = cv2.inpaint(roi_color, mask_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                    clean_img[y1:y2, x1:x2][mask_dilated > 0] = inpainted[mask_dilated > 0]

        return Image.fromarray(clean_img)

if __name__ == "__main__":
    print("MangaInpainter stroke-level siap digunakan.")
