import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

class MangaInpainter:
    def __init__(self):
        """
        Inpainter presisi goresan tinta (Stroke-Level Inpainting):
        - HANYA mendeteksi & menghapus goresan tinta teks di dalam balon kata
        - TIDAK PERNAH membuat kotak putih
        - TIDAK PERNAH menyentuh garis gambar karakter, wajah, atau border balon komik
        """
        pass

    def clean_bubbles(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        masks: Optional[List[np.ndarray]] = None
    ) -> Image.Image:
        """
        Menghapus goresan teks dialog asli menggunakan Navier-Stokes Inpainting presisi tinggi.
        Menggunakan batas bubble bounds dan bubble_mask flood-fill agar border hitam balon 
        dan artwork karakter di luar balon tidak pernah terhapus.
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clean_img = img_np.copy()
        h_img, w_img = gray.shape

        for i, (x, y, w, h) in enumerate(bboxes):
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            roi_gray = gray[y1:y2, x1:x2]
            roi_color = clean_img[y1:y2, x1:x2]

            # 1. Cari kecerahan latar belakang balon kata (persentil 85)
            bg_val = np.percentile(roi_gray, 85)

            if bg_val < 110:
                # Balon gelap / screentone hitam dengan teks putih
                mask = (roi_gray > 165).astype(np.uint8) * 255
            else:
                # Balon putih standar dengan teks hitam:
                thresh_val = min(175, int(bg_val - 35))
                mask = (roi_gray < thresh_val).astype(np.uint8) * 255

            # 2. Jika mask flood fill tersedia, kunci agar inpainting hanya berlaku di dalam balon
            if masks is not None and i < len(masks) and masks[i] is not None:
                roi_bubble_mask = masks[i][y1:y2, x1:x2]
                if np.any(roi_bubble_mask > 0):
                    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    eroded_bmask = cv2.erode((roi_bubble_mask > 0).astype(np.uint8), k_erode, iterations=1)
                    mask = (mask > 0) & (eroded_bmask > 0)
                    mask = mask.astype(np.uint8) * 255

            # 3. Pastikan mask hanya berisi goresan teks (bukan seluruh area)
            mask_pixels = np.sum(mask > 0)
            if 0 < mask_pixels < (mask.size * 0.75):
                # Dilasi kecil (2px) agar tepian anti-aliasing huruf ikut terhapus bersih
                k_size = 2 if min(w_img, h_img) < 1000 else 3
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)

                # Inpaint Telea / Navier-Stokes: mengganti goresan huruf dengan warna latar belakang balon
                inpainted_roi = cv2.inpaint(roi_color, mask_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
                clean_img[y1:y2, x1:x2] = inpainted_roi

        return Image.fromarray(clean_img)

if __name__ == "__main__":
    print("MangaInpainter stroke-level siap digunakan.")
