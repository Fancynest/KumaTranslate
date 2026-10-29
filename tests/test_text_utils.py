import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from src.text_utils import (
    normalize_ocr_text,
    split_sentences,
    apply_glossary_source,
    is_hallucinated_short,
    is_strong_name_candidate,
    assess_translation
)
from src.translator import MangaTranslator


class TestTextUtils(unittest.TestCase):
    def test_normalize_ocr_text(self):
        # Hapus spasi di antara dua karakter kana/Han
        self.assertEqual(normalize_ocr_text("ボ ル ト は 強 い"), "ボルトは強い")
        # Hangul mempertahankan spasi (bahasa Korea)
        self.assertEqual(normalize_ocr_text("안녕 하세요"), "안녕 하세요")
        # Variasi elipsis
        self.assertEqual(normalize_ocr_text("えっ・・・"), "えっ…")
        self.assertEqual(normalize_ocr_text("えっ...."), "えっ…")
        self.assertEqual(normalize_ocr_text("えっ……"), "えっ…")
        self.assertEqual(normalize_ocr_text("えっ‥"), "えっ…")
        # '・' tunggal dipertahankan untuk nama asing
        self.assertEqual(normalize_ocr_text("アルバート・ルイス"), "アルバート・ルイス")
        # Menghapus garis sisa OCR
        self.assertEqual(normalize_ocr_text("|ボルト｜"), "ボルト")
        # Teks non-CJK hanya di-strip
        self.assertEqual(normalize_ocr_text("  Hello world!  "), "Hello world!")

    def test_split_sentences(self):
        # 3 segmen terpisah setelah 。！？
        res1 = split_sentences("こんにちは。元気？うん！")
        self.assertEqual(res1, ["こんにちは。", "元気？", "うん！"])

        # Rangkaian tanda baca '！？' tidak dipecah di tengah
        res2 = split_sentences("えっ！？")
        self.assertEqual(res2, ["えっ！？"])

        # Penutup kutip tidak dipecah sebelumnya
        res3 = split_sentences("「待て！」と言った")
        self.assertEqual(res3, ["「待て！」と言った"])

        # Pecahan < 2 karakter bermakna ditempel ke tetangga
        res4 = split_sentences("あ。元気？")
        self.assertEqual(res4, ["あ。元気？"])

    def test_apply_glossary_source(self):
        # Katakana lookaround: 'ミナ' tidak boleh merusak 'ミナト'
        self.assertEqual(
            apply_glossary_source("ミナトは来た", {"ミナ": "Mina"}),
            "ミナトは来た"
        )
        # Istilah terpanjang diproses dulu
        self.assertEqual(
            apply_glossary_source("ミナトは来た", {"ミナ": "Mina", "ミナト": "Minato"}),
            "Minatoは来た"
        )
        # Kanji replace biasa
        self.assertEqual(
            apply_glossary_source("田中先生", {"田中": "Tanaka"}),
            "Tanaka先生"
        )
        # Hangul replace biasa
        self.assertEqual(
            apply_glossary_source("진우는 갔다", {"진우": "Jinu"}),
            "Jinu는 갔다"
        )

    def test_is_hallucinated_short(self):
        # Terjemahan sah tidak boleh dibuang
        self.assertFalse(is_hallucinated_short("大丈夫？", "Kamu baik-baik saja?"))
        self.assertFalse(is_hallucinated_short("どうも", "Terima kasih banyak."))
        self.assertFalse(is_hallucinated_short("やめろ", "Hentikan!"))

        # Halusinasi khas SFX / onomatopoeia katakana pendek
        self.assertTrue(is_hallucinated_short("グリ…", "Aku tidak tahu."))
        self.assertTrue(is_hallucinated_short("ゴゴゴ", "Aku tidak tahu apa yang terjadi."))

        # n > 3 atau n == 0 tidak dianggap short hallucination
        self.assertFalse(is_hallucinated_short("これはテストです", "Ini adalah sebuah pengujian yang sangat panjang"))
        self.assertFalse(is_hallucinated_short("", "Apa kabar"))

    def test_is_strong_name_candidate(self):
        # before_honorific + auto_accept -> True
        cand1 = {
            "term": "サスケ",
            "auto_accept": True,
            "suggested_translation": "Sasuke",
            "reasons": {"before_honorific"}
        }
        self.assertTrue(is_strong_name_candidate(cand1))

        # Hanya katakana tanpa konteks nama -> False
        cand2 = {
            "term": "ケーキ",
            "auto_accept": True,
            "suggested_translation": "Keki",
            "reasons": ["katakana"]
        }
        self.assertFalse(is_strong_name_candidate(cand2))

        # auto_accept False -> False
        cand3 = {
            "term": "サスケ",
            "auto_accept": False,
            "suggested_translation": "Sasuke",
            "reasons": {"before_honorific"}
        }
        self.assertFalse(is_strong_name_candidate(cand3))

        # Support reasons berupa set atau list
        cand4 = {
            "term": "サラダ",
            "auto_accept": True,
            "suggested_translation": "Sarada",
            "reasons": ["introduced_character", "katakana"]
        }
        self.assertTrue(is_strong_name_candidate(cand4))

    def test_assess_translation(self):
        # Terjemahan kosong
        flags_empty = assess_translation("テスト", "")
        self.assertIn("empty_translation", flags_empty)

        # Kebocoran skrip asli
        flags_leak = assess_translation("テスト", "Ini adalah テスト hasil")
        self.assertIn("source_script_leak", flags_leak)

        # Pemotongan makna (truncation)
        flags_trunc = assess_translation("これはとても長いテキストで文章が続いています", "Pendek")
        self.assertIn("possible_truncation", flags_trunc)

        # Overgeneration pada teks pendek
        flags_over = assess_translation("やめろ", "Jangan lakukan hal itu lagi karena itu sangat berbahaya bagi kita")
        self.assertIn("possible_overgeneration", flags_over)


class TestMockTranslateBatch(unittest.TestCase):
    def test_mock_translate_batch_pipeline(self):
        """
        Memverifikasi alur translate_batch:
        - Normalisasi teks OCR & penguraian glosarium lookaround
        - Pemecahan multi-kalimat & urutan segmen
        - Chunking tepat per 16 kalimat
        - Penggabungan kembali per balon kata
        """
        tr = object.__new__(MangaTranslator)
        tr.device = "cpu"
        tr.default_src_lang = "jpn_Jpan"
        tr.default_tgt_lang = "ind_Latn"
        tr.num_beams = 4
        tr.repetition_penalty = 1.05

        # Buat Mock Tokenizer & Model
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 1234
        
        captured_chunks: List[List[str]] = []

        def fake_tokenize(chunk_texts, **kwargs):
            captured_chunks.append(list(chunk_texts))
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = {"input_ids": [1] * len(chunk_texts)}
            return mock_tensor

        def fake_batch_decode(tokens, **kwargs):
            # Decode segmen dari chunk yang terakhir diproses
            last_chunk = captured_chunks[-1]
            return [f"TR_{seg}" for seg in last_chunk]

        mock_tokenizer.side_effect = fake_tokenize
        mock_tokenizer.batch_decode.side_effect = fake_batch_decode

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()

        tr.tokenizer = mock_tokenizer
        tr.model = mock_model

        # Siapkan 10 input balon kata:
        # - Sebagian berisi multi-kalimat sehingga menghasilkan total 20 segmen (> 16 untuk menguji chunking)
        # - Berisi glosarium untuk menguji lookaround
        glossary = {"ミナ": "Mina", "ミナト": "Minato"}
        input_texts = [
            "こんにちは。元気？",               # 2 segmen
            "ミナトは来た。ミナではない。",      # 2 segmen -> Minatoは来た。, ミナではない。
            "ボ ル ト は 強 い！",               # 1 segmen -> ボルトは強い！
            "",                                  # kosong -> dikembalikan as-is
            "1。", "2。", "3。", "4。", "5。",   # 5 segmen
            "6。", "7。", "8。", "9。", "10。",  # 5 segmen
            "11。", "12。", "13。", "14。", "15。" # 5 segmen
        ]
        # Total segmen bermakna: 2 + 2 + 1 + 0 + 5 + 5 + 5 = 20 segmen!

        results = tr.translate_batch(input_texts, glossary=glossary)

        # 1. Verifikasi chunking 16
        # 20 segmen harus dipecah menjadi 2 chunk: [16 segmen] dan [4 segmen]
        self.assertEqual(len(captured_chunks), 2)
        self.assertEqual(len(captured_chunks[0]), 16)
        self.assertEqual(len(captured_chunks[1]), 4)

        # 2. Verifikasi glosarium lookaround dan urutan segmen
        chunk0 = captured_chunks[0]
        self.assertEqual(chunk0[0], "こんにちは。")
        self.assertEqual(chunk0[1], "元気？")
        self.assertEqual(chunk0[2], "Minatoは来た。")
        self.assertEqual(chunk0[3], "Minaではない。")
        self.assertEqual(chunk0[4], "ボルトは強い！")

        # 3. Verifikasi penggabungan kembali hasil per balon kata
        self.assertEqual(len(results), len(input_texts))
        self.assertEqual(results[0], "TR_こんにちは。 TR_元気？")
        self.assertEqual(results[1], "TR_Minatoは来た。 TR_Minaではない。")
        self.assertEqual(results[2], "TR_ボルトは強い！")
        self.assertEqual(results[3], "")


if __name__ == "__main__":
    unittest.main()
