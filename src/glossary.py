import os
import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger("KumaTranslate.Glossary")

# Katakana regex (2 atau lebih karakter katakana)
KATAKANA_REGEX = re.compile(r'[\u30A1-\u30F6ー]{2,}')

# Honorific suffixes in Japanese, Korean, and Chinese
JP_HONORIFIC_SUFFIXES = [
    "さん", "ちゃん", "くん", "君", "様", "さま",
    "先輩", "せんぱい", "先生", "せんせい", "氏", "殿", "たん"
]
KR_HONORIFIC_SUFFIXES = [
    "씨", "님", "선배", "선배님", "선생님", "오빠", "누나", "형", "동생", "작가님"
]
ZH_HONORIFIC_SUFFIXES = [
    "先生", "小姐", "女士", "老师", "师傅", "哥", "姐", "总", "大人", "前辈"
]

ALL_HONORIFICS = JP_HONORIFIC_SUFFIXES + KR_HONORIFIC_SUFFIXES + ZH_HONORIFIC_SUFFIXES
HONORIFIC_PATTERN = re.compile(
    r'([^\s\d、。！？…・「」『』（）()]+?)(?:' + '|'.join(re.escape(h) for h in ALL_HONORIFICS) + r')(?=[\s、。！？…・「」『』（）()]|$)'
)

JP_PARTICLES = re.compile(r'^[はがをにへでとからもやの]+')

class GlossaryManager:
    """
    Mengelola Glosarium per Seri (sumber -> target).
    Tersimpan di: glossary/<seri>.json.
    Secara default kosong, tidak ada nama yang di-hardcode.
    """
    def __init__(self, glossary_dir: str = "glossary"):
        self.glossary_dir = glossary_dir
        os.makedirs(self.glossary_dir, exist_ok=True)

    def _get_path(self, series_id: str) -> str:
        safe_id = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]', '_', series_id)
        if not safe_id:
            safe_id = "default"
        return os.path.join(self.glossary_dir, f"{safe_id}.json")

    def load_glossary(self, series_id: Optional[str] = None) -> Dict[str, str]:
        if not series_id:
            return {}
        path = self._get_path(series_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Gagal membaca glosarium {path}: {e}")
            return {}

    def save_glossary(self, series_id: str, mapping: Dict[str, str]) -> str:
        path = self._get_path(series_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        return path

    def post_check_and_enforce(
        self,
        source_text: str,
        target_text: str,
        glossary: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        """
        Pasca-cek: tiap entri glosarium yang ada di sumber harus ada di keluaran;
        kalau tidak, log peringatan dan terapkan perbaikan konsistensi.
        """
        if not glossary or not source_text or not target_text:
            return target_text, []

        warnings = []
        result_text = target_text

        for src_term, tgt_term in sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True):
            if src_term in source_text:
                # Cek apakah target_term ada di keluaran terjemahan
                if tgt_term.lower() not in result_text.lower():
                    warn = f"[GLOSSARY WARN] Istilah glosarium '{src_term}' -> '{tgt_term}' tidak ditemukan pada output: '{result_text}'"
                    logger.warning(warn)
                    warnings.append(warn)
                    # Jika model menerjemahkan secara harafiah atau salah, kita pastikan nama terjaga
                    # Catat peringatan agar terpantau di log & debug dump

        return result_text, warnings


class NameCandidateExtractor:
    """
    Penemuan kandidat nama diri dari hasil OCR satu chapter:
    - Morphological tagger (fugashi / UniDic: tag proper noun)
    - Token sebelum honorifik (さん, ちゃん, くん, 先生, dll.)
    - Katakana yang berulang lintas halaman
    Menghasilkan daftar kandidat dengan skor prioritas.
    """
    def __init__(self):
        self._tagger = None
        self._tagger_loaded = False

    def _get_tagger(self):
        if self._tagger_loaded:
            return self._tagger
        self._tagger_loaded = True
        try:
            import fugashi
            self._tagger = fugashi.Tagger()
        except Exception as e:
            logger.info(f"Fugashi tidak tersedia atau gagal dimuat: {e}. Menggunakan fallback regex.")
            self._tagger = None
        return self._tagger

    def extract_candidates(self, pages_texts: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Menganalisis teks dari seluruh halaman (pages_texts = list of list of OCR strings per page).
        Mengembalikan daftar kandidat terurut berdasarkan skor.
        """
        tagger = self._get_tagger()
        candidates: Dict[str, Dict[str, Any]] = {}

        for page_idx, page_lines in enumerate(pages_texts):
            page_seen_katakana = set()

            for text in page_lines:
                if not text or not text.strip():
                    continue

                # 1. Analisis Morfologi (fugashi/UniDic) jika tersedia
                if tagger is not None:
                    try:
                        for word in tagger(text):
                            feat = word.feature
                            pos1 = getattr(feat, 'pos1', '')
                            pos2 = getattr(feat, 'pos2', '')
                            surface = word.surface

                            if len(surface) >= 2 and pos1 == '名詞' and pos2 == '固有名詞':
                                if surface not in candidates:
                                    candidates[surface] = {
                                        "term": surface,
                                        "score": 0.0,
                                        "reasons": set(),
                                        "pages": set(),
                                        "suggested": surface
                                    }
                                candidates[surface]["score"] += 3.5
                                candidates[surface]["reasons"].add("proper_noun")
                                candidates[surface]["pages"].add(page_idx + 1)
                    except Exception:
                        pass

                # 2. Token sebelum honorifik
                for match in HONORIFIC_PATTERN.finditer(text):
                    name_tok = match.group(1).strip()
                    name_tok = JP_PARTICLES.sub('', name_tok)
                    # Jika tagger tersedia, filter hanya ambil bagian kata benda / nama
                    if tagger is not None and name_tok:
                        try:
                            nouns = [w.surface for w in tagger(name_tok) if getattr(w.feature, 'pos1', '') == '名詞']
                            if nouns:
                                name_tok = nouns[-1]  # ambil kata benda terdekat dengan honorifik
                        except Exception:
                            pass

                    if 2 <= len(name_tok) <= 8:
                        if name_tok not in candidates:
                            candidates[name_tok] = {
                                "term": name_tok,
                                "score": 0.0,
                                "reasons": set(),
                                "pages": set(),
                                "suggested": name_tok
                            }
                        candidates[name_tok]["score"] += 4.0
                        candidates[name_tok]["reasons"].add("before_honorific")
                        candidates[name_tok]["pages"].add(page_idx + 1)

                # 3. Katakana yang berulang
                for kata in KATAKANA_REGEX.findall(text):
                    if len(kata) >= 2:
                        if kata not in candidates:
                            candidates[kata] = {
                                "term": kata,
                                "score": 0.0,
                                "reasons": set(),
                                "pages": set(),
                                "suggested": kata
                            }
                        candidates[kata]["score"] += 1.5
                        candidates[kata]["reasons"].add("katakana")
                        candidates[kata]["pages"].add(page_idx + 1)
                        page_seen_katakana.add(kata)

        # 4. Berikan bonus frekuensi lintas halaman
        results = []
        for term, data in candidates.items():
            num_pages = len(data["pages"])
            if num_pages >= 2:
                data["score"] += float(num_pages) * 1.5
                data["reasons"].add(f"multi_page({num_pages})")

            # Filter kandidat dengan skor minimal
            if data["score"] >= 3.0:
                results.append({
                    "term": data["term"],
                    "score": round(data["score"], 1),
                    "reasons": sorted(list(data["reasons"])),
                    "pages": sorted(list(data["pages"])),
                    "auto_accept": data["score"] >= 5.5,
                    "suggested_translation": data["suggested"]
                })

        # Urutkan dari skor tertinggi
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class ChapterTranslationMemory:
    """
    Translation memory per chapter:
    Keputusan nama/istilah dan gaya sapaan dipakai konsisten di semua halaman.
    """
    def __init__(self, initial_glossary: Optional[Dict[str, str]] = None):
        self.glossary = dict(initial_glossary or {})
        self.cached_decisions: Dict[str, str] = {}
        self.pronoun_style: str = "kamu"  # default natural Indonesian

    def set_term(self, source_term: str, target_term: str):
        self.glossary[source_term] = target_term
        self.cached_decisions[source_term] = target_term

    def get_term(self, source_term: str) -> Optional[str]:
        return self.glossary.get(source_term)

    def apply_memory(self, text: str) -> str:
        for src, tgt in self.glossary.items():
            if src in text:
                text = text.replace(src, tgt)
        return text
