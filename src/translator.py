import os
import re
import json
import logging
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("KumaTranslate.Translator")

class LLMTranslator:
    """
    Backend terjemahan berbasis LLM (Ollama lokal atau OpenAI-compatible API).
    Menerjemahkan seluruh balon dalam 1 halaman sekaligus dalam reading order,
    dengan output JSON per nomor balon kata: {"1": "...", "2": "..."}.
    """
    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 20
    ):
        self.api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout

    def translate_bubbles(
        self,
        texts: List[str],
        src_lang: str = "jpn_Jpan",
        tgt_lang: str = "ind_Latn",
        series_hint: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None
    ) -> List[str]:
        if not texts:
            return []

        target_name = "Indonesian (Bahasa Indonesia)" if tgt_lang == "ind_Latn" else "English"
        source_name = "Japanese" if src_lang == "jpn_Jpan" else ("Korean" if src_lang == "kor_Hang" else "Chinese")

        prompt_lines = [f"{i+1}: {t}" for i, t in enumerate(texts)]
        system_prompt = (
            f"You are a professional {source_name}-to-{target_name} manga localization translator and visual dialogue analyst.\n\n"
            "CORE LOCALIZATION PRINCIPLES:\n"
            f"1. Natural Dialogue: Produce fluent, natural, and expressive {target_name} dialogue suitable for manga characters (santai, luwes, hidup, tidak kaku seperti terjemahan mesin harfiah).\n"
            "2. Scene & Context Awareness: Speech bubbles are arranged in manga reading order. Read connected bubbles and the entire page as one continuous scene. Resolve omitted Japanese/Korean subjects and objects (zero pronouns) using scene context and character dynamics without inventing facts.\n"
            "3. Emotion, Tone & Intent: Faithfully preserve character voice, emotion, sarcasm, threats, humor, suspense, politeness, and dramatic emphasis.\n"
            "4. Bubble Space Efficiency: Keep dialogue concise and punchy so it fits comfortably inside comic speech bubbles without text overflow.\n"
            "5. Proper Names & Glossary: NEVER translate proper character names or series titles into literal common nouns (e.g. do not translate names into foods, objects, or common words). Strictly follow romanized names and the glossary if provided.\n"
            "6. SFX & Artwork: Do not translate sound effects (SFX) or onomatopoeia artwork; keep them as original or leave unchanged.\n"
            "7. Strict Accuracy & Invariance: Do not invert negation, tense, or intent. Do not change statements into questions. Do not alter who performs an action. Do not invent explanations or facts absent from the original dialogue.\n\n"
            "OUTPUT FORMAT:\n"
            'Return STRICTLY a single JSON object mapping each bubble number (string) to its localized translation string. '
            'Example: {"1": "...", "2": "..."}. Do not include explanations, thinking tags, or markdown outside the JSON.'
        )

        user_content = f"Series Title Hint: {series_hint or 'None'}\n"
        if glossary:
            user_content += f"Glossary (Source -> Target): {json.dumps(glossary, ensure_ascii=False)}\n"
        user_content += "\nSpeech bubbles in reading order:\n" + "\n".join(prompt_lines)
        user_content += f'\n\nTranslate each numbered bubble into natural {target_name}. Return strictly a JSON object {{"1": "...", "2": "..."}}.'

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        url = f"{self.api_base}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Robust JSON extraction
        if "```" in content:
            content = re.sub(r'```(?:json)?', '', content).strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError(f"Tidak dapat menemukan JSON pada respons LLM: {content[:100]}...")

        parsed_data = json.loads(match.group(0))
        results = []
        for i in range(len(texts)):
            key_str = str(i + 1)
            raw_val = None
            if isinstance(parsed_data, dict):
                raw_val = (
                    parsed_data.get(key_str) 
                    or parsed_data.get(i + 1) 
                    or parsed_data.get(f"bubble_{key_str}") 
                    or parsed_data.get(f"bubble_{i + 1}")
                )
                if raw_val is None and "translations" in parsed_data:
                    sub = parsed_data["translations"]
                    if isinstance(sub, list) and i < len(sub):
                        raw_val = sub[i]
                    elif isinstance(sub, dict):
                        raw_val = sub.get(key_str) or sub.get(f"bubble_{key_str}")
            elif isinstance(parsed_data, list) and i < len(parsed_data):
                raw_val = parsed_data[i]

            if isinstance(raw_val, dict):
                val = (
                    raw_val.get("indonesian_translation")
                    or raw_val.get("translation")
                    or raw_val.get("translated_text")
                    or raw_val.get("text")
                    or ""
                )
            elif isinstance(raw_val, str):
                val = raw_val
            else:
                val = ""
            results.append(str(val).strip())

        return results

class MangaTranslator:
    def __init__(
        self,
        model_name: Optional[str] = None,
        default_src_lang: str = "jpn_Jpan",
        default_tgt_lang: str = "ind_Latn",
        device: Optional[str] = None,
        num_beams: int = 4,
        repetition_penalty: float = 1.05
    ):
        """
        Translator multi-bahasa ultra-cepat menggunakan NLLB-200 dengan akselerasi FP16 (Half Precision).
        Dioptimalkan untuk Tensor Cores RTX 2050 (3x-5x lebih cepat dari FP32).
        
        Evaluasi Model:
        - facebook/nllb-200-distilled-600M:
            * VRAM FP16: ~1.15 GB (Peak ~1.30 GB)
            * Sangat stabil dan muat bersamaan dengan MangaOCR (~1.2 GB) dan YOLOv8 (~0.3 GB) di GPU 4 GB.
        - facebook/nllb-200-distilled-1.3B:
            * VRAM FP16: ~2.56 GB (Peak ~2.78 GB)
            * Muat pada 4 GB GPU saat standalone, namun rentan OOM jika dijalankan paralel penuh bersama MangaOCR + YOLO.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.default_src_lang = default_src_lang
        self.default_tgt_lang = default_tgt_lang
        self.model_name = model_name or os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

        print(f"[TRANSLATOR] Memuat model NLLB-200 ({self.model_name}) pada {self.device.upper()} (FP16, beams={self.num_beams}, rep_penalty={self.repetition_penalty})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=default_src_lang)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.llm = LLMTranslator()
        print(f"[TRANSLATOR] Model NLLB-200 FP16 & LLM Backend siap beroperasi.")

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
        tgt_lang: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Menerjemahkan BANYAK kalimat sekaligus dalam 1 panggilan GPU (Batch Processing).
        Mendukung pra-substitusi glosarium istilah dan nama diri agar tidak diterjemahkan harafiah.
        """
        if not texts:
            return []

        source = src_lang or self.default_src_lang
        target = tgt_lang or self.default_tgt_lang

        if source == target:
            return texts

        # Bersihkan teks input dan terapkan pra-substitusi glosarium
        valid_indices = []
        sub_texts = []
        for i, t in enumerate(texts):
            # Jika hanya titik-titik atau tanda seru, tidak perlu ke model
            if t and t.strip():
                valid_indices.append(i)
                cleaned_input = t.strip()
                if glossary:
                    for s_term, t_term in sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True):
                        if s_term in cleaned_input:
                            cleaned_input = cleaned_input.replace(s_term, t_term)
                sub_texts.append(cleaned_input)

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
                repetition_penalty=self.repetition_penalty,
                num_beams=self.num_beams
            )

        batch_results = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )

        # Petakan kembali dan bersihkan halusinasi
        final_results = list(texts)
        for idx, res_text in zip(valid_indices, batch_results):
            final_results[idx] = self._clean_translated_text(res_text)

        return final_results

    def translate_page_bubbles(
        self,
        texts: List[str],
        backend: str = "auto",
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        series_hint: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None
    ) -> Tuple[List[str], str]:
        """
        Menerjemahkan seluruh balon dalam satu halaman komik.
        Mendukung backend 'llm' (Ollama/API) dengan reading order dan output JSON terstruktur,
        serta otomatis fallback ke NLLB-200 jika LLM gagal atau offline.
        
        Mengembalikan (daftar_terjemahan, backend_yang_digunakan).
        """
        if not texts:
            return [], backend

        chosen_backend = backend.lower() if backend else "auto"

        if chosen_backend in ["llm", "auto"]:
            try:
                # LLM mendukung terjemahan ke Bahasa Indonesia maupun English
                if (tgt_lang or self.default_tgt_lang) in ["ind_Latn", "eng_Latn"]:
                    results = self.llm.translate_bubbles(
                        texts=texts,
                        src_lang=src_lang or self.default_src_lang,
                        tgt_lang=tgt_lang or self.default_tgt_lang,
                        series_hint=series_hint,
                        glossary=glossary
                    )
                    if results and len(results) == len(texts) and any(r.strip() for r in results):
                        return results, "llm"
            except Exception as e:
                warn_msg = f"[LLM FALLBACK] Backend LLM gagal ({type(e).__name__}: {e}). Beralih otomatis ke NLLB-200 offline."
                print(warn_msg)
                logger.warning(warn_msg)

        # Fallback ke NLLB-200 batch translator
        nllb_results = self.translate_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang, glossary=glossary)
        return nllb_results, "nllb"

    def translate(self, text: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None) -> str:
        if not text or not text.strip():
            return ""
        results = self.translate_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        return results[0] if results else ""

if __name__ == "__main__":
    tr = MangaTranslator()
    res = tr.translate_batch(["あれ？", "係船柱！"], src_lang="jpn_Jpan", tgt_lang="ind_Latn")
    print("Hasil:", res)
