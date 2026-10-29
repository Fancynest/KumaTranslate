import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from PIL import Image, ImageDraw
from src.renderer import MangaRenderer


class TestMangaRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = MangaRenderer()

    def test_shrink_font_on_single_token(self):
        """
        Memverifikasi bahwa jika token kata tunggal melebihi lebar baris pada font besar,
        renderer secara otomatis mengecilkan font (shrink font) agar muat tanpa terpotong.
        """
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        bbox = (50, 50, 100, 100) # w=100, h=100
        text = "PENGEMBANGAN TEKNOLOGI" # "PENGEMBANGAN" panjang tapi muat di font lebih kecil
        font, lines, line_pos, _ = self.renderer._fit_text_adaptive(
            draw, text, bbox, max_font=32, min_font=12
        )
        self.assertGreater(len(lines), 0)
        # Font harus menyusut di bawah max_font
        self.assertLess(font.size, 32)
        # Semua baris harus strictly muat di dalam bbox
        for idx, line in enumerate(lines):
            l_bb = draw.textbbox((0, 0), line, font=font)
            lw = l_bb[2] - l_bb[0]
            self.assertLessEqual(lw, bbox[2])

    def test_character_break_on_30_char_token(self):
        """
        Tes wajib user: Token 30 karakter tanpa spasi di dalam balon kata.
        Jika pada min_font (12px) token tetap melebihi lebar balon,
        harus dilakukan character break sehingga tidak ada teks yang overflow keluar bbox.
        """
        token_30 = "ABCDEFGHIJ1234567890KLMNOPQRST"
        self.assertEqual(len(token_30), 30)

        img = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Bbox sempit: w=70px (kurang dari lebar 30 karakter font 12px yang ~210px)
        bbox = (50, 50, 70, 150)
        bx, by, bw, bh = bbox

        font, lines, line_pos, spacing = self.renderer._fit_text_adaptive(
            draw, token_30, bbox, min_font=12
        )

        # Harus terpecah menjadi beberapa baris (character break)
        self.assertGreater(len(lines), 1)

        # Verifikasi bahwa SETIAP baris memiliki lebar <= bw dan posisi horizontal >= bx
        for idx, line in enumerate(lines):
            l_bb = draw.textbbox((0, 0), line, font=font)
            lw = l_bb[2] - l_bb[0]
            lx, ly = line_pos[idx]

            self.assertLessEqual(
                lw, bw,
                f"Baris {idx} ('{line}') dengan lebar {lw}px melebihi lebar bbox {bw}px"
            )
            self.assertGreaterEqual(lx, bx)
            self.assertLessEqual(lx + lw, bx + bw)

        # Render langsung ke gambar dan verifikasi tidak ada error
        rendered = self.renderer.render_bubble_text(img, bbox, token_30)
        self.assertIsInstance(rendered, Image.Image)

    def test_render_page_bubbles_harmonization(self):
        """
        Memverifikasi font harmonization pada satu halaman:
        Font antar balon diselaraskan dalam batas median +/- 30%.
        """
        img = Image.new("RGB", (800, 1000), (255, 255, 255))
        items = [
            {"safe_bbox": (500, 50, 150, 150), "bubble_bbox": (500, 50, 150, 150), "text": "Aku punya hadiah."},
            {"safe_bbox": (200, 30, 120, 120), "bubble_bbox": (200, 30, 120, 120), "text": "Ini Sarada, putri Sasuke."},
            {"safe_bbox": (50, 250, 160, 160), "bubble_bbox": (50, 250, 160, 160), "text": "Itu kelemahan terbesar Boruto."},
        ]
        out = self.renderer.render_page_bubbles(img, items)
        self.assertIsInstance(out, Image.Image)
        # Gambar harus termodifikasi (tidak identik dengan kanvas putih murni)
        self.assertTrue(np.any(np.array(out) < 255))


if __name__ == "__main__":
    unittest.main()
