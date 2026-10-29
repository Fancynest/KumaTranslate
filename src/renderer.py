import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Any

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
        min_font: int = 12
    ) -> Tuple[ImageFont.FreeTypeFont, List[str], List[Tuple[int, int]], int]:
        """
        Menghitung ukuran font maksimal dan layout baris teks adaptif.
        Menyesuaikan kontur alami balon kata (oval, persegi, atau gerigi screaming)
        dengan penempatan teks terpusat secara vertikal dan verifikasi lebar irisan per baris.
        """
        bx, by, bw, bh = bbox
        raw_words = text.split()
        if not raw_words:
            font = ImageFont.truetype(self.font_path, min_font)
            return font, [], [], 1

        # Pecah kata bersambung tanda hubung (cth: 'Makoto-chan', 'Tsubai-san')
        # agar dapat dipisah baris saat lebar balon sempit, menghasilkan ukuran font yang jauh lebih terbaca
        words = []
        for w in raw_words:
            if '-' in w and not w.startswith('-') and not w.endswith('-'):
                parts = w.split('-')
                for idx_p, p in enumerate(parts):
                    if idx_p < len(parts) - 1:
                        words.append(p + '-')
                    else:
                        words.append(p)
            else:
                words.append(w)

        def join_tokens(tokens: List[str]) -> str:
            res = ""
            for t in tokens:
                if not res or res.endswith('-'):
                    res += t
                else:
                    res += " " + t
            return res

        pad_x = max(2, int(bw * 0.08))
        pad_y = max(2, int(bh * 0.08))
        rect_w = max(8, bw - 2 * pad_x)
        rect_h = max(8, bh - 2 * pad_y)

        # Erode mask setebal outline (3px) agar teks terjamin tidak menyentuh outline
        eroded_mask = None
        if mask is not None and np.any(mask > 0):
            k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded_mask = cv2.erode((mask > 0).astype(np.uint8), k_erode, iterations=1)

        def get_line_span(y1: int, y2: int) -> Tuple[int, int]:
            if eroded_mask is None:
                return rect_w, bx + bw // 2
            h_m, w_m = eroded_mask.shape
            cy1 = max(0, min(h_m - 1, int(y1)))
            cy2 = max(0, min(h_m, int(y2)))
            if cy2 <= cy1:
                return rect_w, bx + bw // 2
            slice_rows = eroded_mask[cy1:cy2, :]
            min_span = 999999
            best_cx = bx + bw // 2
            valid = 0
            for r in range(slice_rows.shape[0]):
                pts = np.where(slice_rows[r] > 0)[0]
                if len(pts) > 0:
                    span = pts[-1] - pts[0]
                    if span < min_span:
                        min_span = span
                        best_cx = (pts[0] + pts[-1]) // 2
                    valid += 1
            if valid == 0 or min_span == 999999 or min_span < 12:
                return rect_w, bx + bw // 2
            return max(10, min_span - 6), best_cx

        for size in range(max_font, min_font - 1, -1):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception:
                font = ImageFont.load_default()

            spacing = max(1, int(size * 0.12))
            sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
            line_h = sample_bbox[3] - sample_bbox[1]

            # Aturan: jika ada token tunggal yang lebarnya > rect_w,
            # perkecil font (shrink font if single token > line width)
            token_overflow = False
            for w in words:
                w_bb = draw.textbbox((0, 0), w, font=font)
                if (w_bb[2] - w_bb[0]) > rect_w:
                    token_overflow = True
                    break
            if token_overflow:
                continue

            # 1. Word wrap awal berdasarkan lebar nominal safe area
            lines = []
            cur_line = []
            for w in words:
                cand = join_tokens(cur_line + [w])
                w_bb = draw.textbbox((0, 0), cand, font=font)
                if (w_bb[2] - w_bb[0]) <= rect_w:
                    cur_line.append(w)
                else:
                    if cur_line:
                        lines.append(join_tokens(cur_line))
                        cur_line = [w]
                    else:
                        lines.append(w)
                        cur_line = []
            if cur_line:
                lines.append(join_tokens(cur_line))

            total_h = len(lines) * line_h + (len(lines) - 1) * spacing
            if total_h > rect_h:
                continue

            # 2. Verifikasi posisi vertikal terpusat terhadap kontur mask asli
            start_y = by + max(0, (bh - total_h) // 2)
            line_positions = []
            all_lines_fit = True

            for idx, line in enumerate(lines):
                l_bb = draw.textbbox((0, 0), line, font=font)
                lw = l_bb[2] - l_bb[0]
                y_top = start_y + idx * (line_h + spacing)
                y_bot = y_top + line_h

                avail_w, cx = get_line_span(y_top, y_bot)
                if lw <= avail_w:
                    lx = cx - (lw // 2)
                elif lw <= rect_w:
                    lx = bx + (bw - lw) // 2
                else:
                    all_lines_fit = False
                    break
                line_positions.append((lx, y_top))

            if all_lines_fit:
                return font, lines, line_positions, spacing

        # Jika ada kata panjang yang belum muat pada min_font, coba perkecil hingga batas terbaca (size 7-8)
        # agar kata bahasa Indonesia panjang (cth: 'menyelesaikan', 'mengantuk') tidak terpotong per huruf
        for dyn_size in range(min_font - 1, 6, -1):
            try:
                dyn_font = ImageFont.truetype(self.font_path, dyn_size)
            except Exception:
                dyn_font = ImageFont.load_default()

            dyn_spacing = max(1, int(dyn_size * 0.10))
            sample_bbox = draw.textbbox((0, 0), "Ag", font=dyn_font)
            dyn_line_h = sample_bbox[3] - sample_bbox[1]

            dyn_token_overflow = False
            for w in words:
                w_bb = draw.textbbox((0, 0), w, font=dyn_font)
                if (w_bb[2] - w_bb[0]) > rect_w:
                    dyn_token_overflow = True
                    break
            if dyn_token_overflow:
                continue

            dyn_lines = []
            dyn_cur = []
            for w in words:
                cand = join_tokens(dyn_cur + [w])
                w_bb = draw.textbbox((0, 0), cand, font=dyn_font)
                if (w_bb[2] - w_bb[0]) <= rect_w:
                    dyn_cur.append(w)
                else:
                    if dyn_cur:
                        dyn_lines.append(join_tokens(dyn_cur))
                        dyn_cur = [w]
                    else:
                        dyn_lines.append(w)
                        dyn_cur = []
            if dyn_cur:
                dyn_lines.append(join_tokens(dyn_cur))

            dyn_tot_h = len(dyn_lines) * dyn_line_h + (len(dyn_lines) - 1) * dyn_spacing
            if dyn_tot_h > rect_h:
                continue

            # Verifikasi baris terhadap kontur mask
            start_y = by + max(0, (bh - dyn_tot_h) // 2)
            dyn_line_positions = []
            dyn_all_fit = True
            for idx, line in enumerate(dyn_lines):
                l_bb = draw.textbbox((0, 0), line, font=dyn_font)
                lw = l_bb[2] - l_bb[0]
                y_top = start_y + idx * (dyn_line_h + dyn_spacing)
                y_bot = y_top + dyn_line_h
                avail_w, cx = get_line_span(y_top, y_bot)
                if lw <= avail_w:
                    lx = cx - (lw // 2)
                elif lw <= rect_w:
                    lx = bx + (bw - lw) // 2
                else:
                    dyn_all_fit = False
                    break
                dyn_line_positions.append((lx, y_top))

            if dyn_all_fit:
                return dyn_font, dyn_lines, dyn_line_positions, dyn_spacing

        # Fallback terakhir jika balon sangat sempit: pecah token dengan hyphen (-),
        # dan JANGAN PERNAH menyisakan potongan yatim 1-2 huruf (seperti 'menyelesaika\\nn' atau 'mengantu\\nk')
        font = ImageFont.truetype(self.font_path, max(7, min_font - 4))
        spacing = 1
        sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = sample_bbox[3] - sample_bbox[1]

        broken_words = []
        for w in words:
            w_bb = draw.textbbox((0, 0), w, font=font)
            if (w_bb[2] - w_bb[0]) > rect_w:
                # Potong di suku kata yang rapi dengan minimal 3 karakter per potongan
                cur_chunk = ""
                for idx_ch, ch in enumerate(w):
                    cand_chunk = cur_chunk + ch
                    c_bb = draw.textbbox((0, 0), cand_chunk + "-", font=font)
                    if (c_bb[2] - c_bb[0]) <= rect_w:
                        cur_chunk = cand_chunk
                    else:
                        remaining = len(w) - idx_ch
                        if len(cur_chunk) >= 3 and remaining >= 3:
                            broken_words.append(cur_chunk + "-")
                            cur_chunk = ch
                        elif cur_chunk:
                            cur_chunk += ch
                        else:
                            broken_words.append(ch)
                            cur_chunk = ""
                if cur_chunk:
                    broken_words.append(cur_chunk)
            else:
                broken_words.append(w)

        lines = []
        cur_line = []
        for w in broken_words:
            cand = " ".join(cur_line + [w])
            w_bb = draw.textbbox((0, 0), cand, font=font)
            if (w_bb[2] - w_bb[0]) <= rect_w:
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

        total_h = len(lines) * line_h + (len(lines) - 1) * spacing
        start_y = by + max(0, (bh - total_h) // 2)
        line_positions = []
        for idx, line in enumerate(lines):
            l_bb = draw.textbbox((0, 0), line, font=font)
            lw = l_bb[2] - l_bb[0]
            y_top = start_y + idx * (line_h + spacing)
            avail_w, cx = get_line_span(y_top, y_top + line_h)
            
            if lw <= avail_w:
                lx = cx - (lw // 2)
            else:
                lx = bx + max(0, (bw - lw) // 2)
            # Pastikan teks strictly berada di dalam batas horizontal
            lx = max(bx, min(bx + bw - lw, lx))
            line_positions.append((lx, y_top))

        return font, lines, line_positions, spacing

    def render_bubble_text(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        text: str,
        bubble_mask: Optional[np.ndarray] = None,
        text_color: Tuple[int, int, int] = (15, 15, 15),
        stroke_color: Tuple[int, int, int] = (255, 255, 255),
        target_max_font: Optional[int] = None,
        target_min_font: Optional[int] = None
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

        # Batas maksimal dan minimal font proporsional ukuran balon (min 11-12px)
        max_font = target_max_font if target_max_font is not None else max(14, min(34, int(bh * 0.38)))
        min_font = target_min_font if target_min_font is not None else max(11, min(14, int(min(bw, bh) * 0.12)))

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

    def render_page_bubbles(
        self,
        image: Image.Image,
        bubble_items: List[Dict[str, Any]],
        text_color: Tuple[int, int, int] = (15, 15, 15),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Menyelaraskan ukuran font antar balon kata dalam satu halaman (page-level harmonization):
        1. Menghitung ukuran font optimal awal per balon kata.
        2. Menghitung median ukuran font satu halaman.
        3. Membatasi ukuran font tiap balon pada rentang [0.70 * median, 1.30 * median] (+/- 30%),
           kecuali teks tidak muat sehingga diizinkan mengecil.
        4. Deteksi bug balon luas dengan font kecil: jika area balon luas (>= 6000 px^2)
           namun safe_bbox terlalu sempit (aspect ratio sempit < 0.45) atau font < 0.70 * median,
           periksa safe_bbox terhadap bubble_bbox/mask dan perluas batas render.
        """
        if not bubble_items:
            return image

        valid_items = [b for b in bubble_items if b.get("text") and str(b.get("text")).strip()]
        if not valid_items:
            return image

        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)

        # 1. First Pass: simulasi ukuran font optimal per balon & koreksi safe_bbox sempit pada balon luas
        fitted_info = []
        for item in valid_items:
            text = str(item["text"]).strip()
            s_box = list(item.get("safe_bbox") or item.get("bubble_bbox") or item["bbox"])
            b_box = list(item.get("bubble_bbox") or s_box)
            mask = item.get("bubble_mask")

            area_outer = b_box[2] * b_box[3]
            # Deteksi anomali balon luas dengan safe_bbox sempit:
            # Jika area balon luas (>= 6000 px^2) tetapi lebar safe_bbox sangat sempit (< 0.40 * tinggi atau < 55px)
            # sedangkan bubble_bbox jauh lebih lebar, gunakan bubble_bbox dengan padding moderat
            if area_outer >= 6000 and (s_box[2] < 55 or (s_box[2] / max(1, s_box[3])) < 0.40) and b_box[2] >= 65:
                pad_w = int(b_box[2] * 0.10)
                pad_h = int(b_box[3] * 0.10)
                s_box = [
                    b_box[0] + pad_w,
                    b_box[1] + pad_h,
                    max(20, b_box[2] - 2 * pad_w),
                    max(20, b_box[3] - 2 * pad_h)
                ]

            max_f = max(14, min(34, int(s_box[3] * 0.38)))
            min_f = max(8, min(14, int(min(s_box[2], s_box[3]) * 0.12)))

            font, lines, positions, spacing = self._fit_text_adaptive(
                draw, text, tuple(s_box), mask=mask, max_font=max_f, min_font=min_f
            )

            fitted_info.append({
                "item": item,
                "text": text,
                "render_box": tuple(s_box),
                "mask": mask,
                "base_font_size": font.size,
                "min_font": min_f
            })

        # 2. Hitung median font size halaman
        font_sizes = [f["base_font_size"] for f in fitted_info]
        import statistics
        page_median_font = statistics.median(font_sizes)
        font_floor = max(11, int(round(page_median_font * 0.70)))
        font_ceil = min(36, int(round(page_median_font * 1.30)))

        # 3. Second Pass: Render dengan pembatasan [font_floor, font_ceil]
        for f_data in fitted_info:
            r_box = f_data["render_box"]
            text = f_data["text"]
            mask = f_data["mask"]
            min_f = max(8, f_data["min_font"] - 2)

            # Coba fit dengan batas harmonis (max_font = font_ceil)
            # Jika teks muat di atas font_floor, font akan berada dalam range [font_floor, font_ceil]
            # Jika teks panjang tidak muat di font_floor, diizinkan turun hingga min_f
            font, lines, line_positions, spacing = self._fit_text_adaptive(
                draw, text, r_box, mask=mask, max_font=font_ceil, min_font=min_f
            )

            if not lines or not line_positions:
                continue

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
