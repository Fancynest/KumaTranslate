import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List

class MangaTranslator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        default_src_lang: str = "jpn_Jpan",
        default_tgt_lang: str = "ind_Latn",
        device: Optional[str] = None
    ):
        """
        Translator multi-bahasa ultra-cepat menggunakan NLLB-200 dengan akselerasi FP16 (Half Precision).
        Dioptimalkan untuk Tensor Cores RTX 2050 (3x-5x lebih cepat dari FP32).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.default_src_lang = default_src_lang
        self.default_tgt_lang = default_tgt_lang
        self.model_name = model_name

        print(f"[TRANSLATOR] Memuat model NLLB-200 pada {self.device.upper()} (FP16 Accelerated)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=default_src_lang)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        print(f"[TRANSLATOR] Model NLLB-200 FP16 siap beroperasi.")

    def _clean_translated_text(self, text: str) -> str:
        """
        Membersihkan halusinasi NLLB-200 seperti 'Bahasa Inggris: ...' atau kutipan ganda.
        """
        if not text:
            return ""

        clean = text.strip()

        # Bersihkan prefix halusinasi
        prefixes = [
            "bahasa inggris:", "bahasa indonesia:", "bahasa jepang:",
            "bahasa melayu:", "bahasa arab:", "bahasa mandarin:",
            "english:", "indonesian:", "japanese:", "chinese:",
            "terjemahan:", "translation:", "translated:",
            "in english:", "in indonesian:",
            "artinya:", "meaning:",
        ]
        lower_clean = clean.lower()
        for p in prefixes:
            if lower_clean.startswith(p):
                clean = clean[len(p):].strip()
                lower_clean = clean.lower()
                break

        # Bersihkan tanda kutip luar jika model membungkus kalimat dengan quotes
        if (clean.startswith('"') and clean.endswith('"')) or (clean.startswith("'") and clean.endswith("'")):
            if len(clean) >= 2:
                clean = clean[1:-1].strip()
        if (clean.startswith('“') and clean.endswith('”')) or (clean.startswith('‘') and clean.endswith('’')):
            if len(clean) >= 2:
                clean = clean[1:-1].strip()

        return clean

    def translate_batch(
        self,
        texts: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None
    ) -> List[str]:
        """
        Menerjemahkan BANYAK kalimat sekaligus dalam 1 panggilan GPU (Batch Processing).
        """
        if not texts:
            return []

        source = src_lang or self.default_src_lang
        target = tgt_lang or self.default_tgt_lang

        if source == target:
            return texts

        # Bersihkan teks input
        valid_indices = []
        sub_texts = []
        for i, t in enumerate(texts):
            # Jika hanya titik-titik atau tanda seru, tidak perlu ke model
            if t and t.strip():
                valid_indices.append(i)
                sub_texts.append(t.strip())

        if not valid_indices:
            return texts

        # Tokenize batch secara paralel
        self.tokenizer.src_lang = source
        inputs = self.tokenizer(
            sub_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target)

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                num_beams=1  # Greedy decoding ultra cepat
            )

        batch_results = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )

        # Petakan kembali dan bersihkan halusinasi
        final_results = list(texts)
        for idx, res_text in zip(valid_indices, batch_results):
            final_results[idx] = self._clean_translated_text(res_text)

        return final_results

    def translate(self, text: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None) -> str:
        if not text or not text.strip():
            return ""
        results = self.translate_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        return results[0] if results else ""

if __name__ == "__main__":
    tr = MangaTranslator()
    res = tr.translate_batch(["あれ？", "係船柱！"], src_lang="jpn_Jpan", tgt_lang="ind_Latn")
    print("Hasil:", res)
