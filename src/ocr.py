import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from manga_ocr import MangaOcr
from rapidocr_onnxruntime import RapidOCR

class MultiLingualOCR:
    def __init__(self, force_cpu: bool = False):
        """
        OCR Multi-Bahasa Cerdas untuk Komik:
        - Manga (Jepang): Manga-OCR (Vision Transformer)
        - Manhua (Mandarin) & Western Comics (Inggris): RapidOCR
        - Manhwa (Korea): EasyOCR
        """
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print(f"[OCR] Menginisialisasi OCR Multi-Bahasa pada device: {self.device.upper()}...")

        # 1. Engine khusus Manga Jepang (Sangat akurat membaca kanji + kana vertikal/horizontal)
        self.manga_ocr = MangaOcr(force_cpu=(self.device == "cpu"))

        # 2. Engine untuk Inggris & Mandarin (Cepat & akurat via ONNX)
        self.rapid_ocr = RapidOCR()

        # Engine khusus teks Latin/Inggris dengan model rec PP-OCRv3 English
        self.rapid_ocr_en = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models", "rapidocr_en")
        rec_model = os.path.join(models_dir, "en_PP-OCRv3_rec_infer.onnx")
        rec_keys = os.path.join(models_dir, "en_dict.txt")

        if not (os.path.exists(rec_model) and os.path.exists(rec_keys)):
            try:
                os.makedirs(models_dir, exist_ok=True)
                import requests
                if not os.path.exists(rec_model):
                    print("[OCR] Mengunduh model pengenalan teks Latin (en_PP-OCRv3_rec)...")
                    r_m = requests.get("https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx", timeout=30)
                    if r_m.status_code == 200:
                        with open(rec_model, "wb") as f:
                            f.write(r_m.content)
                if not os.path.exists(rec_keys):
                    r_k = requests.get("https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt", timeout=15)
                    if r_k.status_code == 200:
                        with open(rec_keys, "w", encoding="utf-8") as f:
                            f.write(r_k.text)
            except Exception as e:
                print(f"[OCR] Info: pengunduhan otomatis model Latin gagal ({e}), menggunakan engine default.")

        if os.path.exists(rec_model) and os.path.exists(rec_keys):
            try:
                self.rapid_ocr_en = RapidOCR(rec_model_path=rec_model, rec_keys_path=rec_keys)
                print("[OCR] Engine RapidOCR English (PP-OCRv3 Latin) aktif.")
            except Exception as e:
                print(f"[OCR] Info: Gagal memuat model RapidOCR English ({e}), fallback ke default.")

        # 3. Engine Korea (Lazy loaded agar hemat RAM/VRAM)
        self._korean_ocr = None
        self._english_secondary_reader = None
        print("[OCR] Engine OCR Multi-Bahasa siap digunakan.")

    def _get_korean_ocr(self):
        if self._korean_ocr is None:
            import easyocr
            print("[OCR] Memuat engine OCR Korea (Manhwa)...")
            self._korean_ocr = easyocr.Reader(['ko', 'en'], gpu=(self.device == "cuda"))
        return self._korean_ocr

    def _get_english_ocr_secondary(self):
        if self._english_secondary_reader is None:
            import easyocr
            print("[OCR] Memuat engine OCR Latin sekunder (EasyOCR)...")
            self._english_secondary_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        return self._english_secondary_reader

    def read_text_secondary(self, image_input: Union[str, Image.Image, np.ndarray], lang: str = "en") -> str:
        """
        Fallback engine OCR saat kualitas pembacaan engine utama rendah (ocr_quality < 0.6):
        Untuk Latin/Inggris: mencoba EasyOCR(['en']) sebagai engine kedua.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return ""
            img = Image.open(image_input).convert("RGB")
            img_np = np.array(img)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 and image_input.shape[2] == 3 else image_input)
        else:
            img = image_input.convert("RGB")
            img_np = np.array(img)
        if lang in ["eng_Latn", "en"]:
            try:
                reader = self._get_english_ocr_secondary()
                results = reader.readtext(img_np, detail=0)
                txt = " ".join(results).strip() if results else ""
                if not txt:
                    inv_res = reader.readtext(255 - img_np, detail=0)
                    txt = " ".join(inv_res).strip() if inv_res else ""
                return txt
            except Exception as e:
                print(f"[OCR] Info: Secondary engine EasyOCR gagal: {e}")
                res, _ = self.rapid_ocr(img_np)
                return " ".join([item[1] for item in res]).strip() if res else ""
        return ""

    def read_text(self, image_input: Union[str, Image.Image, np.ndarray], lang_hint: Optional[str] = None) -> str:
        """
        Membaca teks dari balon kata komik secara otomatis:
        Manga Jepang -> Manhua Mandarin / Comic Barat -> Manhwa Korea
        Mendukung semua warna balon: putih, hitam, abu-abu, screentone, dan bergradasi.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"File gambar tidak ditemukan: {image_input}")
            img = Image.open(image_input).convert("RGB")
            img_np = np.array(img)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 and image_input.shape[2] == 3 else image_input)
        else:
            img = image_input.convert("RGB")
            img_np = np.array(img)

        # Petunjuk eksplisit dari user jika ada
        if lang_hint in ["jpn_Jpan", "ja"]:
            txt = self.manga_ocr(img).strip()
            if not txt:
                txt = self.manga_ocr(Image.fromarray(255 - img_np)).strip()
            return txt
        elif lang_hint in ["kor_Hang", "ko"]:
            reader = self._get_korean_ocr()
            results = reader.readtext(img_np, detail=0)
            txt = " ".join(results).strip() if results else ""
            if not txt:
                inv_res = reader.readtext(255 - img_np, detail=0)
                txt = " ".join(inv_res).strip() if inv_res else ""
            return txt
        elif lang_hint in ["eng_Latn", "en"]:
            engine = self.rapid_ocr_en or self.rapid_ocr
            prep_np = img_np
            if img_np.shape[0] < 90:
                scale = min(2.5, 120.0 / max(1, img_np.shape[0]))
                if scale > 1.1:
                    new_w = int(img_np.shape[1] * scale)
                    new_h = int(img_np.shape[0] * scale)
                    prep_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            res, _ = engine(prep_np)
            txt = " ".join([item[1] for item in res]).strip() if res else ""
            avg_score = float(np.mean([item[2] for item in res])) if res else 0.0
            alpha_count = len(re.findall(r'[A-Za-z]', txt))

            # Jika RapidOCR kosong, atau skor rendah (< 0.65), atau huruf sangat sedikit (<= 1 huruf):
            # Coba secondary OCR (EasyOCR) yang sangat tangguh untuk teks pendek Latin
            if not txt or avg_score < 0.65 or alpha_count <= 1:
                sec_txt = self.read_text_secondary(img_np, lang="en")
                sec_alpha = len(re.findall(r'[A-Za-z]', sec_txt)) if sec_txt else 0
                if sec_txt and sec_alpha > alpha_count:
                    txt = sec_txt
                elif not txt:
                    res_inv, _ = engine(255 - prep_np)
                    txt_inv = " ".join([item[1] for item in res_inv]).strip() if res_inv else ""
                    if txt_inv:
                        txt = txt_inv
                    elif sec_txt:
                        txt = sec_txt
            return txt
        elif lang_hint in ["zho_Hans", "zh"]:
            res, _ = self.rapid_ocr(img_np)
            txt = " ".join([item[1] for item in res]).strip() if res else ""
            if not txt:
                res_inv, _ = self.rapid_ocr(255 - img_np)
                txt = " ".join([item[1] for item in res_inv]).strip() if res_inv else ""
            return txt

        # =========================================================
        # AUTO-ROUTING CERDAS:
        # =========================================================

        # 1. Cek teks Latin / Inggris dengan RapidOCR terlebih dahulu.
        # RapidOCR cepat (ONNX), akurat untuk komik Barat/teks Latin, dan tidak berhalusinasi.
        engine_lat = self.rapid_ocr_en or self.rapid_ocr
        prep_np = img_np
        if img_np.shape[0] < 90:
            scale = min(2.5, 120.0 / max(1, img_np.shape[0]))
            if scale > 1.1:
                new_w = int(img_np.shape[1] * scale)
                new_h = int(img_np.shape[0] * scale)
                prep_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        rapid_res, _ = engine_lat(prep_np)
        rapid_text = " ".join([item[1] for item in rapid_res]).strip() if rapid_res else ""
        if not rapid_text:
            inv_res, _ = engine_lat(255 - prep_np)
            if inv_res:
                rapid_text = " ".join([item[1] for item in inv_res]).strip()

        from src.text_utils import count_english_words, calculate_ocr_quality, split_concatenated_words
        prep_split = split_concatenated_words(rapid_text)
        en_words = max(count_english_words(rapid_text), count_english_words(prep_split))
        q = max(calculate_ocr_quality(rapid_text), calculate_ocr_quality(prep_split))
        has_latin = bool(re.search(r'[a-zA-Z]{2,}', rapid_text))

        # Jika RapidOCR mendeteksi kata-kata kamus bahasa Inggris dengan meyakinkan:
        # INI JELAS TEKS LATIN / INGGRIS! JANGAN panggil Manga-OCR agar tidak terjadi halusinasi karakter Jepang!
        if (en_words >= 2 and q >= 0.35) or (en_words >= 1 and q >= 0.5 and len(rapid_text) <= 25 and has_latin):
            return rapid_text

        # 2. Jika bukan teks Inggris yang jelas, cek Manga-OCR untuk teks Jepang
        manga_text = self.manga_ocr(img).strip()
        if not manga_text:
            manga_text = self.manga_ocr(Image.fromarray(255 - img_np)).strip()
        cjk_chars = len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', manga_text))
        kana_chars = len(re.findall(r'[\u3040-\u30ff]', manga_text))
        latin_chars = len(re.findall(r'[a-zA-Z\uFF21-\uFF3A\uFF41-\uFF5A]', manga_text))

        # Hanya terima Manga-OCR jika benar-benar teks Jepang:
        # - Memiliki minimal 2 karakter Kana (Jepang asli hampir selalu punya partikel/hiragana/katakana)
        # - Atau Kanji bersama Kana
        # - Jika hanya Kanji tanpa Kana, JANGAN terima jika RapidOCR sudah menemukan teks Latin yang jelas
        #   (mencegah halusinasi tanggal Manga-OCR seperti １０月２０日 pada border komik Barat)
        is_genuine_japanese = (
            kana_chars >= 2 or
            (kana_chars >= 1 and re.search(r'[\u4e00-\u9fff]', manga_text) is not None) or
            (re.search(r'[\u4e00-\u9fff]', manga_text) is not None and not has_latin) or
            (cjk_chars > latin_chars * 2 and cjk_chars >= 2)
        )

        if is_genuine_japanese:
            return manga_text

        # 3. Cek Manhwa Korea (Hangul) dengan EasyOCR
        reader = self._get_korean_ocr()
        ko_res = reader.readtext(img_np, detail=0)
        ko_text = " ".join(ko_res).strip() if ko_res else ""
        if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', ko_text):
            return ko_text

        # 4. Jika RapidOCR menghasilkan teks non-kosakata (misal nama tokoh unik atau nomor)
        if rapid_text:
            return rapid_text

        # 5. Fallback terakhir
        return manga_text or ko_text

if __name__ == "__main__":
    ocr = MultiLingualOCR()
    print("MultiLingualOCR siap digunakan.")
