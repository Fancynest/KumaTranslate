import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

class MangaRenderer:
    def __init__(self, font_path: str = "C:\\Windows\\Fonts\\comicbd.ttf"):
        """
        Renderer teks komik presisi:
        - Menghitung lebar baris maksimum per baris adaptif dari bubble_mask
        - Memaksimalkan ukuran font di dalam safe_bbox tanpa pernah menyentuh tepi balon
        - Tanpa background kotak, dengan stroke putih tipis agar kontras di atas artwork
        """
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arialbd.ttf"
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
        self.font_path = font_path

    def _compute_line_width_from_mask(
        self,
        mask: np.ndarray,
        y_top: int,
        y_bottom: int,
        margin: int = 4
    ) -> Tuple[int, int]:
        """
        Menghitung lebar horizontal maksimum yang aman dan titik tengah x
        pada irisan vertikal [y_top, y_bottom] dari bubble_mask.
        """
        h_m, w_m = mask.shape
        y1 = max(0, int(y_top))
        y2 = min(h_m, int(y_bottom))
        if y2 <= y1:
            return 0, 0

        slice_rows = mask[y1:y2, :]
        min_span = 999999
        best_center = 0
        valid_rows = 0

        for r in range(slice_rows.shape[0]):
            row_pixels = np.where(slice_rows[r] > 0)[0]
            if len(row_pixels) > 0:
                x_min = int(row_pixels[0])
                x_max = int(row_pixels[-1])
                span = x_max - x_min
                if span < min_span:
                    min_span = span
                    best_center = (x_min + x_max) // 2
                valid_rows += 1

        if valid_rows == 0 or min_span == 999999:
            return 0, 0

        safe_w = max(0, min_span - 2 * margin)
        return safe_w, best_center

    def _fit_text_adaptive(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
        max_font: int = 34,
        min_font: int = 6
    ) -> Tuple[ImageFont.FreeTypeFont, List[str], List[Tuple[int, int]], int]:
        """
        Menghitung ukuran font maksimal dan layout baris teks adaptif.
        Jika bubble_mask diberikan, lebar tiap baris dihitung dari kontur oval balon kata.
        """
        bx, by, bw, bh = bbox
        words = text.split()
        if not words:
            font = ImageFont.truetype(self.font_path, min_font)
            return font, [], [], 1

        pad_x = max(2, int(bw * 0.08))
        pad_y = max(2, int(bh * 0.08))
        rect_w = max(8, bw - 2 * pad_x)
        rect_h = max(8, bh - 2 * pad_y)

        # Erode mask setebal outline (3px) agar teks terjamin tidak menyentuh outline
        eroded_mask = None
        if mask is not None and np.any(mask > 0):
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.erode((mask > 0).astype(np.uint8), k_erode, iterations=1)

        for size in range(max_font, min_font - 1, -1):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                font = ImageFont.load_default()

            spacing = max(1, int(size * 0.12))
            sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
            line_h = sample_bbox[3] - sample_bbox[1]

            # 1. Coba word wrap baris demi baris
            lines = []
            cur_line = []
            fits = True

            # Estimasi awal pembagian baris
            for w in words:
                cand = " ".join(cur_line + [w])
                w_bb = draw.textbbox((0, 0), cand, font=font)
                cand_w = w_bb[2] - w_bb[0]

                # Perkirakan posisi baris saat ini
                est_line_idx = len(lines)
                est_y_top = by + est_line_idx * (line_h + spacing)
                est_y_bot = est_y_top + line_h

                if eroded_mask is not None:
                    avail_w, _ = self._compute_line_width_from_mask(eroded_mask, est_y_top, est_y_bot, margin=3)
                    if avail_w <= 0:
                        avail_w = rect_w
                else:
                    avail_w = rect_w

                if cand_w <= avail_w:
                    cur_line.append(w)
                else:
                    if cur_line:
                        lines.append(" ".join(cur_line))
                        cur_line = [w]
                    else:
                        # Kata tunggal lebih lebar dari ruang baris
                        fits = False
                        break

            if not fits:
                continue

            if cur_line:
                lines.append(" ".join(cur_line))

            total_h = len(lines) * line_h + (len(lines) - 1) * spacing
            if total_h > rect_h:
                continue

            # 2. Verifikasi ulang posisi vertikal dan batas horizontal tiap baris
            start_y = by + max(0, (bh - total_h) // 2)
            line_positions = []
            verified = True

            for idx, line in enumerate(lines):
                l_bb = draw.textbbox((0, 0), line, font=font)
                lw = l_bb[2] - l_bb[0]
                y_top = start_y + idx * (line_h + spacing)
                y_bot = y_top + line_h

                if eroded_mask is not None:
                    avail_w, cx = self._compute_line_width_from_mask(eroded_mask, y_top, y_bot, margin=3)
                    if avail_w > 0 and lw <= avail_w:
                        cur_x = cx - (lw // 2)
                    elif lw <= rect_w:
                        cur_x = bx + max(0, (bw - lw) // 2)
                    else:
                        verified = False
                        break
                else:
                    if lw <= rect_w:
                        cur_x = bx + max(0, (bw - lw) // 2)
                    else:
                        verified = False
                        break

                line_positions.append((cur_x, y_top))

            if verified:
                return font, lines, line_positions, spacing

        # Fallback ukuran terkecil
        font = ImageFont.truetype(self.font_path, min_font)
        spacing = 1
        sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = sample_bbox[3] - sample_bbox[1]

        lines = []
        cur_line = ""
        for char in text:
            test = cur_line + char
            tb = draw.textbbox((0, 0), test, font=font)
            if (tb[2] - tb[0]) <= rect_w:
                cur_line = test
            else:
                if cur_line:
                    lines.append(cur_line)
                    cur_line = char
                else:
                    lines.append(char)
                    cur_line = ""
        if cur_line:
            lines.append(cur_line)

        total_h = len(lines) * line_h + (len(lines) - 1) * spacing
        start_y = by + max(0, (bh - total_h) // 2)
        line_positions = []
        for idx, line in enumerate(lines):
            l_bb = draw.textbbox((0, 0), line, font=font)
            lw = l_bb[2] - l_bb[0]
            cur_x = bx + max(0, (bw - lw) // 2)
            cur_y = start_y + idx * (line_h + spacing)
            line_positions.append((cur_x, cur_y))

        return font, lines, line_positions, spacing

    def render_bubble_text(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        text: str,
        bubble_mask: Optional[np.ndarray] = None,
        text_color: Tuple[int, int, int] = (15, 15, 15),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Render teks terjemahan ke dalam balon kata (safe_bbox).
        - Menghitung lebar maksimum per baris dari bubble_mask
        - Memaksimalkan ukuran font di dalam safe_bbox tanpa menyentuh tepi balon kata.
        """
        if not text or not text.strip():
            return image

        clean_text = text.strip()
        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)
        bx, by, bw, bh = bbox

        # Batas maksimal font proporsional ukuran balon (hingga 34px)
        max_font = max(10, min(34, int(bh * 0.38)))
        min_font = 6

        font, lines, line_positions, spacing = self._fit_text_adaptive(
            draw, clean_text, bbox, mask=bubble_mask, max_font=max_font, min_font=min_font
        )

        if not lines or not line_positions:
            return image

        stroke_w = 1 if font.size < 14 else 2

        for line, (lx, ly) in zip(lines, line_positions):
            draw.text(
                (lx, ly),
                line,
                fill=text_color,
                font=font,
                stroke_width=stroke_w,
                stroke_fill=stroke_color
            )

        return img_out

if __name__ == "__main__":
    print("MangaRenderer siap digunakan.")
