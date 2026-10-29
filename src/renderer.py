import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

class MangaRenderer:
    def __init__(self, font_path: str = "C:\\Windows\\Fonts\\comicbd.ttf"):
        """
        Renderer teks komik:
        - Teks dijamin muat di dalam area teks asli (text_bbox)
        - Auto-fit ukuran font dan word wrapping
        - Tanpa background kotak, dengan stroke putih tipis agar terbaca di atas sisa artwork
        """
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arialbd.ttf"
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
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
        Cari ukuran font dan pembagian baris yang muat di dalam target_w x target_h.
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

            spacing = max(1, int(size * 0.12))

            # Cek apakah ada kata tunggal yang lebih lebar dari target
            too_wide = False
            for w in words:
                wb = draw.textbbox((0, 0), w, font=font)
                if (wb[2] - wb[0]) > target_w:
                    too_wide = True
                    break
            if too_wide:
                continue

            # Word wrap
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

            # Cek apakah muat secara vertikal
            line_boxes = [draw.textbbox((0, 0), l, font=font) for l in lines]
            total_h = sum([b[3] - b[1] for b in line_boxes]) + (len(lines) - 1) * spacing

            if total_h <= target_h:
                return font, lines, spacing

        # Fallback: font terkecil, wrap per karakter
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
        bbox: Tuple[int, int, int, int],
        text: str,
        text_color: Tuple[int, int, int] = (15, 15, 15),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Render teks terjemahan ke dalam balon kata (safe_bbox).
        Strict fitting menjamin teks muat rapi di dalam rongga balon tanpa meluber keluar.
        """
        if not text or not text.strip():
            return image

        clean_text = text.strip()
        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)
        bx, by, bw, bh = bbox

        # Inner padding 10% agar teks tidak mepet border lengkung balon
        pad_x = max(2, int(bw * 0.10))
        pad_y = max(2, int(bh * 0.10))
        target_w = max(8, bw - 2 * pad_x)
        target_h = max(8, bh - 2 * pad_y)

        # Font proporsional terhadap ukuran rongga balon:
        # Dibatasi maksimal 26px agar frasa pendek ("Batang kapal") tidak menjadi raksasa
        max_font = max(8, min(26, int(bh * 0.22)))
        min_font = 6

        font, lines, spacing = self._fit_text_to_box(
            draw, clean_text, target_w, target_h, max_font=max_font, min_font=min_font
        )

        if not lines:
            return image

        line_boxes = [draw.textbbox((0, 0), l, font=font) for l in lines]
        line_heights = [b[3] - b[1] for b in line_boxes]
        total_text_h = sum(line_heights) + (len(lines) - 1) * spacing

        # Posisi vertikal: rata tengah
        cur_y = by + max(0, (bh - total_text_h) // 2)
        stroke_w = 1 if font.size < 14 else 2

        for line, lh, lb in zip(lines, line_heights, line_boxes):
            lw = lb[2] - lb[0]
            # Posisi horizontal: rata tengah
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
    print("MangaRenderer siap digunakan.")
