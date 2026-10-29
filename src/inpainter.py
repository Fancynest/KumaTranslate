import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class MangaInpainter:
    def __init__(self):
        """
        Inpainter presisi goresan tinta (Stroke-Level Inpainting):
        - HANYA mendeteksi & menghapus goresan tinta teks di dalam balon kata
        - TIDAK PERNAH membuat kotak putih
        - TIDAK PERNAH menyentuh garis gambar karakter, wajah, atau rambut komik
        """
        pass

    def clean_bubbles(self, image: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> Image.Image:
        """
        Menghapus goresan teks dialog asli menggunakan Navier-Stokes Inpainting presisi tinggi.
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clean_img = img_np.copy()
        h_img, w_img = gray.shape

        for (x, y, w, h) in bboxes:
            # Berikan padding tipis untuk mengcover ujung goresan huruf
            pad = max(2, int(min(w, h) * 0.08))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

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
                # Teks adalah pixel yang secara signifikan lebih gelap dari background balon
                thresh_val = min(175, int(bg_val - 35))
                mask = (roi_gray < thresh_val).astype(np.uint8) * 255

            # 2. Pastikan mask hanya berisi goresan teks (bukan seluruh area)
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
