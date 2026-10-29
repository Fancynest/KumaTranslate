import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from src.text_utils import (
    count_english_words,
    is_english_word,
    postprocess_nllb_indonesian,
    calculate_ocr_quality
)
from src.lang_detector import detect_document_language, detect_language


class TestEnglishRoutingAndTranslation(unittest.TestCase):
    """
    Regression test suite: Memastikan komik berbahasa Inggris tidak rusak
    oleh halusinasi MangaOCR atau salah deteksi bahasa dokumen.
    """

    def test_english_vocab_helpers(self):
        """Memverifikasi fungsi penghitungan kata kamus bahasa Inggris."""
        self.assertTrue(is_english_word("morning"))
        self.assertTrue(is_english_word("island"))
        self.assertTrue(is_english_word("people"))
        self.assertFalse(is_english_word("xyznonexistent"))

        sentence = "THIS IS ONE OF THE ISLANDS IN THE SETO INLAND SEA."
        self.assertGreaterEqual(count_english_words(sentence), 8)

    def test_detect_document_language_english_with_hallucinations(self):
        """
        Bahkan jika OCR sempat menghasilkan 1-2 karakter Jepang sisa/halusinasi,
        dokumen dengan mayoritas teks Inggris HARUS dideteksi sebagai 'eng_Latn'.
        """
        english_bubbles = [
            "This is one of the islands in the Seto inland sea. It's home for 300 people.",
            "It's called Futaiwa island.",
            "It's been three months since I first moved in.",
            "I came all the way from Tokyo to work on a community-building project.",
            "I work as a staff in this newly build diner called 'Futaiwa Terrace'.",
            "I need to live here for a year to complete the project.",
            "Good morning Makoto-chan.",
            "You're early today.",
            "Aren't you sleepy waking up so early?",
            "Good morning.",
            "Its ok. I'm getting used to this.",
            "Oh my, you're becoming more of a country boy."
        ]
        # Sisipkan satu teks dengan 1 karakter kana sisa
        english_bubbles_with_noise = english_bubbles + ["RNNAC 6002Mの"]

        lang = detect_document_language(english_bubbles_with_noise)
        self.assertEqual(lang, "eng_Latn", "Dokumen mayoritas Inggris harus terdeteksi sebagai eng_Latn")

    def test_detect_document_language_japanese_manga(self):
        """Memverifikasi bahwa manga Jepang asli tetap terdeteksi sebagai 'jpn_Jpan'."""
        japanese_bubbles = [
            "手土産を持ってきてやったぞ",
            "こいつはサラダ…うちはサスケの娘にして",
            "あのボルトにとっての最大の弱みだ",
            "僕の知る限りではな……",
            "認めたくはないが正直ボルトは手強い…",
            "この娘の事となると様子が変わるんだ"
        ]
        lang = detect_document_language(japanese_bubbles)
        self.assertEqual(lang, "jpn_Jpan", "Manga Jepang harus terdeteksi sebagai jpn_Jpan")

    def test_postprocess_english_preserves_names(self):
        """
        Jika src_lang adalah 'eng_Latn', nama/kata 'Bolt' TIDAK boleh diubah menjadi 'Boruto',
        dan kata makanan 'salad' TIDAK boleh diubah menjadi 'Sarada'.
        """
        en_text = "He tightened the Bolt on the machine and ate a fresh salad."
        processed = postprocess_nllb_indonesian(en_text, src_lang="eng_Latn")
        self.assertIn("Bolt", processed)
        self.assertIn("salad", processed)
        self.assertNotIn("Boruto", processed)
        self.assertNotIn("Sarada", processed)

    def test_postprocess_english_refines_manga_dialogue(self):
        """Koreksi frase dialog bahasa Inggris yang diterjemahkan harfiah oleh NLLB."""
        self.assertEqual(
            postprocess_nllb_indonesian("kamu pagi hari ini.", src_lang="eng_Latn"),
            "kamu datang lebih awal hari ini."
        )
        self.assertEqual(
            postprocess_nllb_indonesian("dia anak negara.", src_lang="eng_Latn"),
            "dia anak desa."
        )


if __name__ == "__main__":
    unittest.main()
