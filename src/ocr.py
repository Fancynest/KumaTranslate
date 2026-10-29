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
        elif lang_hint in ["eng_Latn", "en"]:
            engine = self.rapid_ocr_en or self.rapid_ocr
            res, _ = engine(img_np)
            return " ".join([item[1] for item in res]).strip() if res else ""
        elif lang_hint in ["zho_Hans", "zh"]:
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
        engine_lat = self.rapid_ocr_en or self.rapid_ocr
        rapid_res, _ = engine_lat(img_np)
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
