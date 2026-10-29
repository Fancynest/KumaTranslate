import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from PIL import Image
from src.translator import VLMTranslator, MangaTranslator

class TestVLMBackend(unittest.TestCase):
    def test_vlm_base64_conversion(self):
        img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        url = VLMTranslator._image_to_base64_url(img, max_dim=100)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(url), 50)

    def test_vlm_fallback_to_nllb(self):
        """
        Memverifikasi bahwa jika backend VLM offline atau tidak dapat dijangkau,
        sistem secara otomatis dan aman fallback ke NLLB-200 offline tanpa melempar exception crash.
        """
        # Arahkan ke port lokal yang sengaja tidak ada server
        vlm = VLMTranslator(api_base="http://127.0.0.1:59999/v1", timeout=1)
        crops = [Image.new("RGB", (64, 64), color=(255, 255, 255))]
        texts = ["手土産を持ってきてやったぞ"]

        # Panggil translate_page_bubbles dengan backend="vlm"
        # Buat MangaTranslator dummy/mock atau gunakan model yang sudah diinisialisasi
        translator = MangaTranslator(device="cuda")
        translator.vlm = vlm # Paksa gunakan endpoint yang offline

        results, used_backend = translator.translate_page_bubbles(
            texts=texts,
            backend="vlm",
            bubble_crops=crops,
            page_image=crops[0]
        )

        # Harus otomatis fallback ke nllb
        self.assertEqual(used_backend, "nllb")
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0].strip()), 0)

if __name__ == "__main__":
    unittest.main()
