import os
import re
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

        # 3. Engine Korea (Lazy loaded agar hemat RAM/VRAM)
        self._korean_ocr = None
        print("[OCR] Engine OCR Multi-Bahasa siap digunakan.")

    def _get_korean_ocr(self):
        if self._korean_ocr is None:
            import easyocr
            print("[OCR] Memuat engine OCR Korea (Manhwa)...")
            self._korean_ocr = easyocr.Reader(['ko', 'en'], gpu=(self.device == "cuda"))
        return self._korean_ocr

    def read_text(self, image_input: Union[str, Image.Image], lang_hint: Optional[str] = None) -> str:
        """
        Membaca teks dari balon kata komik secara otomatis:
        Manga Jepang -> Manhua Mandarin / Comic Barat -> Manhwa Korea
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"File gambar tidak ditemukan: {image_input}")
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        img_np = np.array(img)

        # Petunjuk eksplisit dari user jika ada
        if lang_hint in ["jpn_Jpan", "ja"]:
            return self.manga_ocr(img).strip()
        elif lang_hint in ["kor_Hang", "ko"]:
            reader = self._get_korean_ocr()
            results = reader.readtext(img_np, detail=0)
            return " ".join(results).strip()
        elif lang_hint in ["eng_Latn", "en", "zho_Hans", "zh"]:
            res, _ = self.rapid_ocr(img_np)
            return " ".join([item[1] for item in res]).strip() if res else ""

        # =========================================================
        # AUTO-ROUTING CERDAS:
        # =========================================================

        # 1. Prioritaskan Manga-OCR untuk teks Jepang
        manga_text = self.manga_ocr(img).strip()
        # Jika mengandung Hiragana atau Katakana, 100% adalah Bahasa Jepang (Manga)
        if re.search(r'[\u3040-\u30ff]', manga_text):
            return manga_text

        # 2. Cek RapidOCR untuk Teks Latin (Inggris) atau Hanzi Mandarin (Manhua)
        rapid_res, _ = self.rapid_ocr(img_np)
        if rapid_res:
            rapid_text = " ".join([item[1] for item in rapid_res]).strip()
            # Jika mengandung karakter Hanzi Mandarin atau kata Latin (Inggris)
            if re.search(r'[\u4e00-\u9fff]', rapid_text) or re.search(r'[a-zA-Z]{2,}', rapid_text):
                return rapid_text

        # 3. Cek Manhwa Korea (Hangul) dengan EasyOCR
        reader = self._get_korean_ocr()
        ko_res = reader.readtext(img_np, detail=0)
        ko_text = " ".join(ko_res).strip() if ko_res else ""
        if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', ko_text):
            return ko_text

        # 4. Fallback jika tidak ada trigger khusus
        return manga_text or ko_text

if __name__ == "__main__":
    ocr = MultiLingualOCR()
    print("MultiLingualOCR siap digunakan.")
