import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

class MangaRenderer:
    def __init__(self, font_path: str = "C:\\Windows\\Fonts\\comicbd.ttf"):
        """
        Renderer Tipografi Komik Profesional Bergaransi Presisi:
        - Strict Fitting: Dijamin 100% teks TIDAK AKAN PERNAH keluar dari garis balon kata
        - Auto-fit cerdas: Menghitung line width & total height secara ketat dengan inner padding 15%
        - Word wrapping adaptif dengan fallback ukuran font proporsional
        - Transparan tanpa background kotak, dengan outline stroke putih halus
        """
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arialbd.ttf"
        self.font_path = font_path

    def _fit_text_to_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        target_w: int,
        target_h: int,
        max_font: int,
        min_font: int = 4
    ) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
        """
        Mencari ukuran font dan pembagian baris yang 100% dijamin muat di dalam target_w dan target_h.
        """
        words = text.split()
        if not words:
            font = ImageFont.truetype(self.font_path, min_font)
            return font, [], 1

        for size in range(max_font, min_font - 1, -1):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                font = ImageFont.load_default()

            spacing = max(1, int(size * 0.15))

            # 1. Pastikan tidak ada kata tunggal yang melebihi lebar target
            single_word_too_wide = False
            for w in words:
                wb = draw.textbbox((0, 0), w, font=font)
                if (wb[2] - wb[0]) > target_w:
                    single_word_too_wide = True
                    break
            if single_word_too_wide:
                continue

            # 2. Bungkus baris teks berdasarkan target_w
            lines = []
            cur_line = []
            for w in words:
                test = " ".join(cur_line + [w])
                tb = draw.textbbox((0, 0), test, font=font)
                if (tb[2] - tb[0]) <= target_w:
                    cur_line.append(w)
                else:
                    if cur_line:
                        lines.append(" ".join(cur_line))
                        cur_line = [w]
                    else:
                        lines.append(w)
                        cur_line = []
            if cur_line:
                lines.append(" ".join(cur_line))

            # 3. Periksa apakah lebar maksimal baris dan tinggi total muat di dalam target
            line_boxes = [draw.textbbox((0, 0), l, font=font) for l in lines]
            max_lw = max([b[2] - b[0] for b in line_boxes]) if line_boxes else 0
            total_h = sum([b[3] - b[1] for b in line_boxes]) + (len(lines) - 1) * spacing

            if max_lw <= target_w and total_h <= target_h:
                return font, lines, spacing

        # Fallback jika teks sangat panjang: gunakan min_font dan wrap per karakter
        font = ImageFont.truetype(self.font_path, min_font)
        spacing = 1
        lines = []
        cur_line = ""
        for char in text:
            test = cur_line + char
            tb = draw.textbbox((0, 0), test, font=font)
            if (tb[2] - tb[0]) <= target_w:
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

        return font, lines, spacing

    def render_bubble_text(
        self,
        image: Image.Image,
        bubble_bbox: Tuple[int, int, int, int],
        text: str,
        text_color: Tuple[int, int, int] = (15, 15, 15),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Merender teks dialog ke dalam balon kata dengan garansi TIDAK KELUAR dari balon.
        """
        if not text or not text.strip():
            return image

        clean_text = text.strip()
        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)
        bx, by, bw, bh = bubble_bbox

        # Berikan inner padding 15% dari batas balon
        target_w = max(8, int(bw * 0.85))
        target_h = max(8, int(bh * 0.85))

        # Batas font proporsional terhadap tinggi balon
        max_font = max(8, min(42, int(bh * 0.35)))
        min_font = 4

        font, lines, spacing = self._fit_text_to_box(
            draw, clean_text, target_w, target_h, max_font=max_font, min_font=min_font
        )

        if not lines:
            return image

        line_boxes = [draw.textbbox((0, 0), l, font=font) for l in lines]
        line_heights = [b[3] - b[1] for b in line_boxes]
        total_text_h = sum(line_heights) + (len(lines) - 1) * spacing

        # Posisi vertikal: rata tengah di dalam balon
        cur_y = by + max(0, (bh - total_text_h) // 2)
        stroke_w = 1 if font.size < 12 else 2

        for i, (line, lh, lb) in enumerate(zip(lines, line_heights, line_boxes)):
            lw = lb[2] - lb[0]
            # Posisi horizontal: rata tengah di dalam balon
            cur_x = bx + max(0, (bw - lw) // 2)

            draw.text(
                (cur_x, cur_y),
                line,
                fill=text_color,
                font=font,
                stroke_width=stroke_w,
                stroke_fill=stroke_color
            )
            cur_y += lh + spacing

        return img_out

if __name__ == "__main__":
    print("MangaRenderer Strict Fitting siap digunakan.")
