import os
import re
import json
import base64
import io
import logging
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List, Dict, Any, Tuple
from src.text_utils import normalize_ocr_text, apply_glossary_source, split_sentences, postprocess_nllb_indonesian

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
        if src_lang in ["jpn_Jpan", "ja"]:
            source_name = "Japanese"
        elif src_lang in ["kor_Hang", "ko"]:
            source_name = "Korean"
        elif src_lang in ["zho_Hans", "zh"]:
            source_name = "Chinese"
        elif src_lang in ["eng_Latn", "en"]:
            source_name = "English"
        else:
            source_name = "Original Comic Language"

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
            "7. Strict Accuracy & Invariance: Do not invert negation, tense, or intent. Do not change statements into questions. Do not alter who performs an action. Do not invent explanations or facts absent from the original dialogue.\n"
            "8. OCR Robustness & Error Invariance: The source dialogue may contain OCR glitches such as words glued together without spaces (e.g. 'YOUKNOW', 'AREMANY'), character misrecognitions/typos (e.g. '1'/'l', 'rn'/'m', '0'/'o', 'teh'/'the'), or hyphenated words broken across line wraps ('SIGNA- TURE'). You must intelligently infer the intended coherent sentence from context before translating.\n\n"
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

class VLMTranslator:
    """
    Backend terjemahan berbasis Vision-Language Model (VLM).
    Menerima crop balon kata (dalam urutan baca) beserta gambar halaman utuh sebagai konteks visual.
    Mengembalikan JSON per nomor balon kata: {"1": "...", "2": "..."}.
    Instruksi khusus: SFX tidak diterjemahkan dan nama diri dipertahankan.
    """
    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 35
    ):
        self.api_base = api_base or os.getenv("VLM_API_BASE", os.getenv("LLM_API_BASE", "http://localhost:11434/v1"))
        self.model = model or os.getenv("VLM_MODEL", "llava:latest")
        self.api_key = api_key or os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", ""))
        self.timeout = timeout

    @staticmethod
    def _image_to_base64_url(img: Image.Image, max_dim: int = 1024) -> str:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def translate_bubbles_vision(
        self,
        texts: List[str],
        bubble_crops: List[Image.Image],
        page_image: Optional[Image.Image] = None,
        src_lang: str = "jpn_Jpan",
        tgt_lang: str = "ind_Latn",
        series_hint: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None
    ) -> List[str]:
        if not texts or not bubble_crops:
            return []

        target_name = "Indonesian (Bahasa Indonesia)" if tgt_lang == "ind_Latn" else "English"
        if src_lang in ["jpn_Jpan", "ja"]:
            source_name = "Japanese"
        elif src_lang in ["kor_Hang", "ko"]:
            source_name = "Korean"
        elif src_lang in ["zho_Hans", "zh"]:
            source_name = "Chinese"
        elif src_lang in ["eng_Latn", "en"]:
            source_name = "English"
        else:
            source_name = "Comic Source Language"

        system_prompt = (
            f"You are a professional visual manga localization translator and image dialogue analyst specializing in {source_name}-to-{target_name}.\n\n"
            "CRITICAL VISUAL LOCALIZATION RULES:\n"
            f"1. Natural Dialogue: Translate dialogue into natural, expressive, conversational {target_name} suitable for manga speech bubbles.\n"
            "2. Reading Order & Visual Context: Bubbles are ordered sequentially (1 to N) following manga layout. Use the full page image to verify who is speaking, facial expressions, tone, and character dynamics.\n"
            "3. NO SFX TRANSLATION: Do NOT translate sound effects (SFX, onomatopoeia, visual sounds such as 'グリ', 'ゴゴ', 'ドカン'). Return an empty string \"\" for any SFX crop to preserve original comic artwork.\n"
            "4. Proper Names & Titles: Do not translate character names literally. Follow established names and the glossary if provided.\n"
            "5. Invariance: Do not invert negation or tense. Do not change questions into statements.\n"
            "6. Output Format: Return STRICTLY a single JSON object mapping each bubble number (string) to translated text: {\"1\": \"...\", \"2\": \"...\"}. Do not include markdown codeblocks or commentary."
        )

        user_content_blocks: List[Dict[str, Any]] = []
        user_content_blocks.append({
            "type": "text",
            "text": f"Series Hint: {series_hint or 'None'}\nGlossary: {json.dumps(glossary or {}, ensure_ascii=False)}\n\nHere is the full page context image:"
        })

        if page_image is not None:
            user_content_blocks.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_base64_url(page_image, max_dim=1024)}
            })

        user_content_blocks.append({
            "type": "text",
            "text": "Below are the individual speech bubble crops in sequential manga reading order (1 to N). Translate each dialogue bubble into natural target language. Remember: DO NOT translate SFX (return \"\" for SFX)."
        })

        for i, (txt, crop) in enumerate(zip(texts, bubble_crops)):
            user_content_blocks.append({
                "type": "text",
                "text": f"Bubble #{i+1} (OCR draft: '{txt}'):"
            })
            if crop is not None:
                user_content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_to_base64_url(crop, max_dim=400)}
                })

        user_content_blocks.append({
            "type": "text",
            "text": f"Return strictly a JSON object mapping bubble numbers to translated dialogue: {{\"1\": \"...\", \"2\": \"...\"}}."
        })

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_blocks}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        url = f"{self.api_base}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        if "```" in content:
            content = re.sub(r'```(?:json)?', '', content).strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError(f"Tidak dapat menemukan JSON pada respons VLM: {content[:100]}...")

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
            elif isinstance(parsed_data, list) and i < len(parsed_data):
                raw_val = parsed_data[i]

            if isinstance(raw_val, dict):
                val = (
                    raw_val.get("translation")
                    or raw_val.get("indonesian_translation")
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
        self.vlm = VLMTranslator()
        print(f"[TRANSLATOR] Model NLLB-200 FP16, LLM Backend, & VLM Backend siap beroperasi.")

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
        Menerjemahkan BANYAK kalimat sekaligus dalam panggilan GPU chunking (Batch Processing).
        Pipeline per teks:
        normalize_ocr_text -> apply_glossary_source -> split_sentences -> chunk 16 NLLB -> reassemble.
        """
        if not texts:
            return []

        source = src_lang or self.default_src_lang
        target = tgt_lang or self.default_tgt_lang

        if source == target:
            return texts

        # 1. Pipeline normalisasi, pra-substitusi glosarium aman, dan pemecahan kalimat
        balloon_segment_slices: List[Tuple[Optional[int], Optional[int], str]] = []
        all_segments: List[str] = []

        for t in texts:
            norm_t = normalize_ocr_text(t) if t else ""
            if not norm_t:
                # Teks yang kosong setelah normalisasi dikembalikan apa adanya
                balloon_segment_slices.append((None, None, t))
                continue

            with_glossary = apply_glossary_source(norm_t, glossary or {})
            segments = split_sentences(with_glossary)

            start = len(all_segments)
            all_segments.extend(segments)
            end = len(all_segments)
            balloon_segment_slices.append((start, end, t))

        if not all_segments:
            return list(texts)

        # 2. Kirim semua segmen ke NLLB per chunk 16 (agar beam search num_beams=4 aman di GPU 4 GB)
        CHUNK_SIZE = 16
        self.tokenizer.src_lang = source
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target)
        translated_segments: List[str] = []

        for c_idx in range(0, len(all_segments), CHUNK_SIZE):
            chunk = all_segments[c_idx:c_idx + CHUNK_SIZE]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

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
            for res_text in batch_results:
                translated_segments.append(self._clean_translated_text(res_text))

        # 3. Gabungkan kembali per balon dengan spasi
        final_results: List[str] = []
        for start, end, orig in balloon_segment_slices:
            if start is None:
                final_results.append(orig)
            else:
                segs = translated_segments[start:end]
                combined = " ".join([s for s in segs if s.strip()])
                # Pasca-pemrosesan khusus Indonesia: perbaiki pronoun, nama, kosakata NLLB
                if target == "ind_Latn" and combined:
                    combined = postprocess_nllb_indonesian(combined, src_lang=source, src_text=orig)
                final_results.append(combined)

        return final_results

    def translate_page_bubbles(
        self,
        texts: List[str],
        backend: str = "auto",
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        series_hint: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None,
        ocr_qualities: Optional[List[float]] = None,
        bubble_crops: Optional[List[Any]] = None,
        page_image: Optional[Any] = None
    ) -> Tuple[List[str], str]:
        """
        Menerjemahkan seluruh balon dalam satu halaman komik.
        Mendukung backend 'llm' (Ollama/API) dan 'vlm' (Vision-Language Model dengan crop visual),
        dengan reading order dan output JSON terstruktur.
        Menjadikan LLM/VLM sebagai jalur utama untuk balon dengan ocr_quality rendah (< 0.6)
        atau sumber sangat pendek (<= 3 kata / karakter), serta otomatis fallback ke NLLB-200 offline.
        
        Mengembalikan (daftar_terjemahan, backend_yang_digunakan).
        """
        if not texts:
            return [], backend

        chosen_backend = backend.lower() if backend else "auto"

        # Normalisasi teks sebelum proses penerjemahan (baik VLM, LLM maupun NLLB)
        normalized_texts = [normalize_ocr_text(t) if t else "" for t in texts]

        # 1. Jalur Backend VLM (Vision-Language Model dengan crop balon & konteks halaman)
        if chosen_backend == "vlm":
            try:
                if bubble_crops and len(bubble_crops) == len(texts):
                    vlm_res = self.vlm.translate_bubbles_vision(
                        texts=normalized_texts,
                        bubble_crops=bubble_crops,
                        page_image=page_image,
                        src_lang=src_lang or self.default_src_lang,
                        tgt_lang=tgt_lang or self.default_tgt_lang,
                        series_hint=series_hint,
                        glossary=glossary
                    )
                    if vlm_res and len(vlm_res) == len(texts) and any(r.strip() for r in vlm_res):
                        return vlm_res, "vlm"
            except Exception as e:
                warn_msg = f"[VLM FALLBACK] Backend VLM gagal ({type(e).__name__}: {e}). Beralih otomatis ke NLLB-200 offline."
                print(warn_msg)
                logger.warning(warn_msg)

        # 2. Jalur Backend LLM (Ollama lokal / OpenAI-compatible API)
        # Periksa apakah ada balon berkualitas rendah atau teks sangat pendek yang membutuhkan inferensi konteks LLM
        has_low_quality = bool(ocr_qualities and any(q < 0.6 for q in ocr_qualities))
        has_short_sources = any(len(t.strip().split()) <= 3 or len(t.strip()) <= 3 for t in texts if t)
        
        should_prioritize_llm = chosen_backend in ["llm", "auto"] and (chosen_backend == "llm" or has_low_quality or has_short_sources)

        if should_prioritize_llm or chosen_backend in ["llm", "auto"]:
            try:
                # LLM mendukung terjemahan ke Bahasa Indonesia maupun English
                if (tgt_lang or self.default_tgt_lang) in ["ind_Latn", "eng_Latn"]:
                    results = self.llm.translate_bubbles(
                        texts=normalized_texts,
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
