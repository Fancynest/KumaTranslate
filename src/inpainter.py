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
        Menghapus goresan teks dialog asli menggunakan Navier-Stokes Inpainting.
        Bekerja pada area text_bbox (lokasi teks asli), bukan seluruh balon.
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clean_img = img_np.copy()
        h_img, w_img = gray.shape

        for i, (x, y, w, h) in enumerate(bboxes):
            # Padding kecil untuk menutupi ujung goresan huruf
            pad = max(2, int(min(w, h) * 0.10))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            if x2 <= x1 or y2 <= y1:
                continue

            roi_gray = gray[y1:y2, x1:x2]
            roi_color = clean_img[y1:y2, x1:x2]

            # 1. Tentukan kecerahan latar belakang (persentil 85)
            bg_val = np.percentile(roi_gray, 85)

            if bg_val < 110:
                # Balon gelap / screentone hitam dengan teks putih
                text_mask = (roi_gray > 165).astype(np.uint8) * 255
            else:
                # Balon putih standar dengan teks hitam
                thresh_val = min(175, int(bg_val - 35))
                text_mask = (roi_gray < thresh_val).astype(np.uint8) * 255

            # 2. Jika bubble mask tersedia, batasi inpainting ke dalam balon saja
            if masks is not None and i < len(masks) and masks[i] is not None:
                roi_bubble = masks[i][y1:y2, x1:x2]
                if np.any(roi_bubble > 0):
                    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    inner_mask = cv2.erode((roi_bubble > 0).astype(np.uint8), k_erode, iterations=1)
                    text_mask = cv2.bitwise_and(text_mask, text_mask, mask=inner_mask)

            # 3. Pastikan mask hanya berisi goresan teks (bukan seluruh area)
            mask_pixels = np.sum(text_mask > 0)
            if 0 < mask_pixels < (text_mask.size * 0.80):
                # Dilasi kecil agar tepian anti-aliasing huruf ikut terhapus
                k_size = 2 if min(w_img, h_img) < 1000 else 3
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)

                inpainted = cv2.inpaint(roi_color, mask_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                clean_img[y1:y2, x1:x2] = inpainted

        return Image.fromarray(clean_img)

if __name__ == "__main__":
    print("MangaInpainter stroke-level siap digunakan.")
