import os
import sys
import argparse
import time
from PIL import Image

# Pastikan console Windows mendukung output UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.lang_detector import detect_language, get_language_name
from src.translator import MangaTranslator
from src.inpainter import MangaInpainter
from src.renderer import MangaRenderer

class MangaTranslationPipeline:
    def __init__(self):
        """
        Orkestrator Lengkap Manga/Manhwa/Manhua Translator:
        - Auto-Detect Bahasa Sumber (Jepang, Korea, Mandarin, Inggris, dll.)
        - Terjemahan ke Bahasa Indonesia & Inggris
        - Inpainting & Typesetting
        """
        print("=" * 65)
        print("MEMULAI PIPELINE TRANSLATE KOMIK (AUTO-DETECT MULTI-BAHASA)")
        print("=" * 65)
        t0 = time.time()

        self.detector = BubbleDetector()
        self.inpainter = MangaInpainter()
        self.renderer = MangaRenderer()

        # Inisialisasi model AI pada GPU CUDA
        self.ocr = MultiLingualOCR()
        self.translator = MangaTranslator()

        print(f"Semua model siap dalam {time.time() - t0:.2f} detik!\n")

    def translate_page(self, image_path: str, target_lang: str = "ind_Latn", output_path: str = None) -> str:
        """
        Menerjemahkan 1 halaman komik dengan opsi target_lang:
        - 'ind_Latn' (Bahasa Indonesia)
        - 'eng_Latn' (Bahasa Inggris)
        - 'both' (Menghasilkan 2 file: versi Indonesia dan versi Inggris)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

        print(f"[PROSES] Menerjemahkan halaman komik: {image_path}")
        img = Image.open(image_path).convert("RGB")

        # 1. Deteksi Balon Kata
        print("[1/5] Mendeteksi letak balon kata...")
        bubbles = self.detector.detect(img)
        print(f"      Ditemukan {len(bubbles)} balon kata.")

        if not bubbles:
            print("      Tidak ada balon kata terdeteksi. Menyimpan salinan gambar asli.")
            if output_path is None:
                output_path = os.path.join("output", os.path.basename(image_path))
            img.save(output_path)
            return output_path

        # 2 & 3. OCR, Auto-Detect Bahasa, dan Terjemahkan
        translated_bubbles_id = []
        translated_bubbles_en = []
        bboxes_to_clean = []

        print("[2/5 & 3/5] OCR, Auto-Detect Bahasa Sumber, dan Menerjemahkan...")
        for i, b in enumerate(bubbles):
            bbox = b["bbox"]
            crop = b["crop"]

            # OCR (Otomatis router Manga/Manhwa/Manhua)
            raw_text = self.ocr.read_text(crop)
            if not raw_text:
                continue

            # Auto-Detect Bahasa
            detected_src = detect_language(raw_text)
            lang_label = get_language_name(detected_src)

            # Terjemahkan ke Indonesia & Inggris
            trans_id = self.translator.translate(raw_text, src_lang=detected_src, tgt_lang="ind_Latn")
            trans_en = self.translator.translate(raw_text, src_lang=detected_src, tgt_lang="eng_Latn")

            print(f"      Balon #{i+1} [Posisi: {bbox}]:")
            print(f"        Teks Asli   : {raw_text}")
            print(f"        Auto-Detect : {lang_label} ({detected_src})")
            print(f"        -> Indonesia: {trans_id}")
            print(f"        -> Inggris  : {trans_en}")

            translated_bubbles_id.append({"bbox": bbox, "text": trans_id})
            translated_bubbles_en.append({"bbox": bbox, "text": trans_en})
            bboxes_to_clean.append(bbox)

        # 4. Hapus Teks Asli (Inpainting)
        print("[4/5] Menghapus teks asli di dalam balon kata...")
        cleaned_img = self.inpainter.clean_bubbles(img, bboxes_to_clean)

        # 5. Pasang Teks Baru (Typesetting / Rendering)
        print("[5/5] Memasang teks terjemahan ke dalam balon kata...")
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        outputs_generated = []

        # Render versi Indonesia
        if target_lang in ["ind_Latn", "both", "id"]:
            final_id = cleaned_img.copy()
            for item in translated_bubbles_id:
                final_id = self.renderer.render_bubble_text(final_id, item["bbox"], item["text"])
            
            path_id = output_path or os.path.join("output", f"{base_name}_indonesia.png")
            final_id.save(path_id)
            print(f"      [OK] Versi Indonesia disimpan ke: {path_id}")
            outputs_generated.append(path_id)

        # Render versi Inggris
        if target_lang in ["eng_Latn", "both", "en"]:
            final_en = cleaned_img.copy()
            for item in translated_bubbles_en:
                final_en = self.renderer.render_bubble_text(final_en, item["bbox"], item["text"])

            path_en = os.path.join("output", f"{base_name}_english.png")
            final_en.save(path_en)
            print(f"      [OK] Versi Inggris disimpan ke: {path_en}")
            outputs_generated.append(path_en)

        print("\n[SELESAI] Proses penerjemahan selesai!")
        return outputs_generated

def main():
    parser = argparse.ArgumentParser(description="Multi-Lingual Comic Translation Pipeline")
    parser.add_argument("--image", type=str, default="input/sample_manga_page.png", help="Path gambar komik input")
    parser.add_argument("--lang", type=str, default="both", help="Bahasa target: 'ind_Latn' (Indonesia), 'eng_Latn' (Inggris), atau 'both' (dua-duanya)")
    args = parser.parse_args()

    pipeline = MangaTranslationPipeline()
    pipeline.translate_page(args.image, target_lang=args.lang)

if __name__ == "__main__":
    main()
