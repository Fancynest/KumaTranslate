import os
import sys
import uuid
import time
import json
import shutil
import string
import re
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PUNCT_REGEX = re.compile(r'[\s\d' + re.escape(string.punctuation) + r'、。！？…・～〜「」『』（）()【】[\]―ー\-–—.,:;!?\'"“”‘’．，]+')

# Setup encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.detector import BubbleDetector
from src.ocr import MultiLingualOCR
from src.lang_detector import detect_language, detect_document_language, get_language_name
from src.translator import MangaTranslator
from src.inpainter import MangaInpainter
from src.renderer import MangaRenderer
from src.document_processor import DocumentProcessor
from src.glossary import GlossaryManager, NameCandidateExtractor, ChapterTranslationMemory
from src.text_utils import (
    normalize_ocr_text,
    is_strong_name_candidate,
    is_hallucinated_short,
    assess_translation,
    post_process_latin_ocr,
    calculate_ocr_quality,
    filter_cross_bubble_hallucinations,
    CJK_REGEX
)
from pydantic import BaseModel

app = FastAPI(title="Manga Translator AI Local", version="2.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model engine (singleton)
class TranslationEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("[ENGINE] Menginisialisasi Model AI (GPU CUDA FP16 Batch Engine)...")
            cls._instance = TranslationEngine()
        return cls._instance

    def __init__(self):
        self.detector = BubbleDetector()
        self.inpainter = MangaInpainter()
        self.renderer = MangaRenderer()
        self.ocr = MultiLingualOCR()
        self.translator = MangaTranslator()
        self.glossary_mgr = GlossaryManager()
        self.name_extractor = NameCandidateExtractor()
        print("[ENGINE] Seluruh model AI & Glosarium berhasil dimuat.")

    def process_single_page(
        self,
        img_path: str,
        target_lang: str = "ind_Latn",
        doc_lang_hint: Optional[str] = None,
        save_debug: bool = True,
        debug_dir: str = "debug_output",
        translate_free_text: bool = False,
        series_id: Optional[str] = None,
        translation_memory: Optional[ChapterTranslationMemory] = None,
        glossary: Optional[Dict[str, str]] = None,
        translation_backend: str = "auto"
    ) -> Dict[str, Any]:
        """
        Memproses 1 halaman secara BATCH PARALEL Presisi Tinggi:
        - Deteksi balon kata (Normalisasi skala cerdas + Filter banner judul)
        - Filter sebelum translate (lewati < 2 char, lewati free_text/SFX secara default)
        - OCR dengan Manga-OCR / RapidOCR
        - Batch translation ke GPU NLLB-200 FP16 dengan guard halusinasi (<= 3 char -> >= 3 kata)
        - Stroke-level inpainting (0% merusak gambar karakter)
        - Strict Fitting Typesetting (100% DIJAMIN teks TIDAK KELUAR dari balon kata)
        - Debug dump JSON per halaman untuk pemisahan diagnosis OCR vs Terjemahan
        """
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size

        # Jika gambar sangat kecil (thumbnail < 650px), tingkatkan resolusi agar font terbaca tajam & muat di dalam balon
        min_dim = min(w_orig, h_orig)
        if min_dim < 650:
            scale_up = 750.0 / min_dim
            new_w = int(w_orig * scale_up)
            new_h = int(h_orig * scale_up)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        bubbles = self.detector.detect(img)

        if not bubbles:
            empty_dump = {
                "image": os.path.basename(img_path),
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "total_bubbles": 0,
                "bubbles": []
            }
            if save_debug:
                os.makedirs(debug_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(img_path))[0]
                with open(os.path.join(debug_dir, f"{stem}_debug.json"), "w", encoding="utf-8") as f:
                    json.dump(empty_dump, f, indent=2, ensure_ascii=False)

            return {
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "images": {"ind_Latn": img, "eng_Latn": img},
                "debug_dump": empty_dump
            }

        # Muat glosarium aktif untuk halaman ini lebih awal
        active_glossary = dict(glossary or {})
        if series_id and not active_glossary:
            active_glossary = self.glossary_mgr.load_glossary(series_id)
        if translation_memory:
            active_glossary.update(translation_memory.glossary)
        glossary_terms = set(active_glossary.keys())

        # 1. OCR teks dan Filter sebelum translate
        raw_candidates = []
        for b in bubbles:
            # Lewati type=="free_text" secara default (kecuali translate_free_text=True) agar tidak menimpa SFX
            if not translate_free_text and b.get("type") == "free_text":
                continue
            raw = self.ocr.read_text(b["crop"], lang_hint=doc_lang_hint)
            if raw and raw.strip():
                raw_candidates.append((b, raw.strip()))

        # Deteksi apakah halaman didominasi teks Latin/Inggris untuk analisis frekuensi kata
        all_words_on_page: Dict[str, int] = {}
        for _, r_txt in raw_candidates:
            for w in re.findall(r'[A-Za-z]+', r_txt):
                lw = w.lower()
                all_words_on_page[lw] = all_words_on_page.get(lw, 0) + 1

        raw_texts = []
        valid_bubbles = []

        for b, raw in raw_candidates:
            is_latin = (
                doc_lang_hint in ["eng_Latn", "en"] or
                (not CJK_REGEX.search(raw) and bool(re.search(r'[A-Za-z]{2,}', raw)))
            )

            if is_latin:
                # Pasca-OCR Latin: sambung tanda hubung, pecah kata menempel, SymSpell, dan hitung ocr_quality
                t_clean, quality = post_process_latin_ocr(
                    raw,
                    chapter_counts=all_words_on_page,
                    glossary_terms=glossary_terms
                )
                if quality < 0.6:
                    # Coba ulang dengan engine kedua (EasyOCR English)
                    raw_sec = self.ocr.read_text_secondary(b["crop"], lang="en")
                    if raw_sec and raw_sec.strip():
                        t_sec, q_sec = post_process_latin_ocr(
                            raw_sec.strip(),
                            chapter_counts=all_words_on_page,
                            glossary_terms=glossary_terms
                        )
                        if q_sec > quality:
                            t_clean, quality = t_sec, q_sec

                b["ocr_quality"] = quality
                b["ocr_low_quality"] = bool(quality < 0.6)
            else:
                t_clean = normalize_ocr_text(raw)
                b["ocr_quality"] = 1.0
                b["ocr_low_quality"] = False

            # Lewati teks yang setelah dibuang spasi dan tanda baca kurang dari 2 karakter
            clean_no_punct = PUNCT_REGEX.sub("", t_clean)
            if len(clean_no_punct) < 2:
                continue

            raw_texts.append(t_clean)
            valid_bubbles.append(b)

        if not raw_texts:
            empty_dump = {
                "image": os.path.basename(img_path),
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "total_bubbles": 0,
                "bubbles": []
            }
            if save_debug:
                os.makedirs(debug_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(img_path))[0]
                with open(os.path.join(debug_dir, f"{stem}_debug.json"), "w", encoding="utf-8") as f:
                    json.dump(empty_dump, f, indent=2, ensure_ascii=False)

            return {
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "images": {"ind_Latn": img, "eng_Latn": img},
                "debug_dump": empty_dump
            }

        # 2. Tentukan bahasa dokumen secara holistik
        if doc_lang_hint:
            page_lang = doc_lang_hint
        else:
            page_lang = detect_document_language(raw_texts)

        # 3. BATCH TRANSLATE (Semua balon sekaligus dalam 1 urutan baca & GPU)
        res_id_texts = []
        res_en_texts = []
        all_warnings = []
        used_backends = {}

        # Muat glosarium aktif untuk halaman ini
        active_glossary = dict(glossary or {})
        if series_id and not active_glossary:
            active_glossary = self.glossary_mgr.load_glossary(series_id)
        if translation_memory:
            active_glossary.update(translation_memory.glossary)

        # Ekstraksi otomatis kandidat nama diri & istilah Katakana/Hangul agar tidak diterjemahkan harafiah
        if page_lang in ["jpn_Jpan", "kor_Hang"]:
            try:
                extracted_names = self.name_extractor.extract_candidates([raw_texts])
                for cand in extracted_names:
                    # Lewati kandidat yang bukan is_strong_name_candidate (kandidat lemah hanya tampil di UI untuk konfirmasi)
                    if not is_strong_name_candidate(cand):
                        continue
                    term = cand.get("term")
                    sugg = cand.get("suggested_translation")
                    if term and sugg and term not in active_glossary:
                        active_glossary[term] = sugg
                        if translation_memory:
                            translation_memory.set_term(term, sugg)
            except Exception as e:
                print(f"[GLOSSARY] Info: ekstraksi nama otomatis dilewati: {e}")

        ocr_q_list = [b.get("ocr_quality", 1.0) for b in valid_bubbles]
        crops_list = [b.get("crop") for b in valid_bubbles]

        if target_lang in ["ind_Latn", "both"]:
            raw_id, backend_id = self.translator.translate_page_bubbles(
                texts=raw_texts,
                backend=translation_backend,
                src_lang=page_lang,
                tgt_lang="ind_Latn",
                series_hint=series_id,
                glossary=active_glossary,
                ocr_qualities=ocr_q_list,
                bubble_crops=crops_list,
                page_image=img
            )
            used_backends["ind_Latn"] = backend_id

            # Guard cerdas halusinasi pada teks pendek
            for src, tgt in zip(raw_texts, raw_id):
                if is_hallucinated_short(src, tgt):
                    print(f"[FILTER] Guard halusinasi membuang: '{src}' -> '{tgt}'")
                    res_id_texts.append("")
                else:
                    tgt_enforced, warns = self.glossary_mgr.post_check_and_enforce(src, tgt, active_glossary)
                    if warns:
                        all_warnings.extend(warns)
                    res_id_texts.append(tgt_enforced)

            # Deteksi halusinasi lintas-balon (mode-collapse: terjemahan >= 3 kata sama dari sumber berbeda)
            res_id_texts = filter_cross_bubble_hallucinations(list(zip(raw_texts, res_id_texts)))

        if target_lang in ["eng_Latn", "both"]:
            raw_en, backend_en = self.translator.translate_page_bubbles(
                texts=raw_texts,
                backend=translation_backend if translation_backend in ["vlm", "llm", "nllb"] else "auto",
                src_lang=page_lang,
                tgt_lang="eng_Latn",
                series_hint=series_hint if 'series_hint' in locals() else series_id,
                glossary=active_glossary,
                ocr_qualities=ocr_q_list,
                bubble_crops=crops_list,
                page_image=img
            )
            used_backends["eng_Latn"] = backend_en

            for src, tgt in zip(raw_texts, raw_en):
                if is_hallucinated_short(src, tgt):
                    print(f"[FILTER] Guard halusinasi membuang: '{src}' -> '{tgt}'")
                    res_en_texts.append("")
                else:
                    tgt_enforced, warns = self.glossary_mgr.post_check_and_enforce(src, tgt, active_glossary)
                    if warns:
                        all_warnings.extend(warns)
                    res_en_texts.append(tgt_enforced)

            # Deteksi halusinasi lintas-balon (mode-collapse: terjemahan >= 3 kata sama dari sumber berbeda)
            res_en_texts = filter_cross_bubble_hallucinations(list(zip(raw_texts, res_en_texts)))

        # 4. Inpainting: Hapus teks asli hanya pada balon yang memiliki terjemahan valid
        #    Jika terjemahan kosong / dibuang karena halusinasi, atau ocr_low_quality,
        #    gambar asli dipertahankan tanpa ditimpa teks cacat/hibrida.
        active_bubbles = []
        for idx, b in enumerate(valid_bubbles):
            t_id = res_id_texts[idx] if idx < len(res_id_texts) else ""
            t_en = res_en_texts[idx] if idx < len(res_en_texts) else ""
            if b.get("ocr_low_quality"):
                print(f"[OCR] Balon #{idx+1} dibiarkan asli karena ocr_low_quality ({b.get('ocr_quality', 0):.2f})")
                continue
            if (target_lang in ["ind_Latn", "both"] and t_id.strip()) or (target_lang == "eng_Latn" and t_en.strip()):
                active_bubbles.append((b, t_id, t_en))

        text_bboxes = [ab[0]["text_bbox"] for ab in active_bubbles]
        bubble_masks = [ab[0].get("bubble_mask") for ab in active_bubbles]
        cleaned_img = self.inpainter.clean_bubbles(img, text_bboxes, masks=bubble_masks) if active_bubbles else img

        # 5. Typesetting: Render teks terjemahan ke dalam safe_bbox (rongga dalam balon kata)
        output_images = {}
        if target_lang in ["ind_Latn", "both"]:
            img_id = cleaned_img.copy()
            items_id = [
                {
                    "safe_bbox": b.get("safe_bbox", b.get("bubble_bbox", b["bbox"])),
                    "bubble_bbox": b.get("bubble_bbox", b["bbox"]),
                    "bubble_mask": b.get("bubble_mask"),
                    "text": t_id
                }
                for b, t_id, _ in active_bubbles if t_id.strip()
            ]
            output_images["ind_Latn"] = self.renderer.render_page_bubbles(img_id, items_id)

        if target_lang in ["eng_Latn", "both"]:
            img_en = cleaned_img.copy()
            items_en = [
                {
                    "safe_bbox": b.get("safe_bbox", b.get("bubble_bbox", b["bbox"])),
                    "bubble_bbox": b.get("bubble_bbox", b["bbox"]),
                    "bubble_mask": b.get("bubble_mask"),
                    "text": t_en
                }
                for b, _, t_en in active_bubbles if t_en.strip()
            ]
            output_images["eng_Latn"] = self.renderer.render_page_bubbles(img_en, items_en)

        # 6. Susun Debug Dump JSON (bubble_id, bbox, text_bbox, type, source, ocr_text, translation)
        debug_dump = {
            "image": os.path.basename(img_path),
            "detected_lang": page_lang,
            "series_id": series_id,
            "translation_backend": used_backends,
            "total_bubbles": len(valid_bubbles),
            "glossary_warnings": all_warnings,
            "bubbles": []
        }
        for idx, b in enumerate(valid_bubbles):
            t_id = res_id_texts[idx] if idx < len(res_id_texts) else ""
            t_en = res_en_texts[idx] if idx < len(res_en_texts) else ""
            s_text = raw_texts[idx] if idx < len(raw_texts) else ""
            q_id = assess_translation(s_text, t_id)
            q_en = assess_translation(s_text, t_en)
            if b.get("ocr_low_quality"):
                q_id.append("ocr_low_quality")
                q_en.append("ocr_low_quality")

            debug_dump["bubbles"].append({
                "bubble_id": idx + 1,
                "bbox": list(b.get("safe_bbox", b.get("bubble_bbox", b.get("bbox")))),
                "text_bbox": list(b.get("text_bbox", b.get("bbox"))),
                "type": b.get("type", "text_bubble"),
                "source": b.get("source", "heuristic"),
                "ocr_text": s_text,
                "ocr_quality": round(float(b.get("ocr_quality", 1.0)), 3),
                "ocr_low_quality": bool(b.get("ocr_low_quality", False)),
                "translation": {
                    "ind_Latn": t_id,
                    "eng_Latn": t_en
                },
                "quality_flags": {
                    "ind_Latn": q_id,
                    "eng_Latn": q_en
                }
            })

        debug_dump["low_quality_bubbles"] = [
            {
                "bubble_id": b["bubble_id"],
                "ocr_text": b["ocr_text"],
                "ocr_quality": b["ocr_quality"]
            }
            for b in debug_dump["bubbles"] if b.get("ocr_low_quality")
        ]

        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            debug_json_path = os.path.join(debug_dir, f"{stem}_debug.json")
            with open(debug_json_path, "w", encoding="utf-8") as f:
                json.dump(debug_dump, f, indent=2, ensure_ascii=False)

        return {
            "detected_lang": page_lang,
            "images": output_images,
            "debug_dump": debug_dump
        }

    # Alias untuk kompatibilitas
    def process_single_image(self, img_path: str, target_lang: str = "ind_Latn"):
        res = self.process_single_page(img_path, target_lang=target_lang)
        return res["images"]

# Task storage in-memory
TASKS: Dict[str, Dict[str, Any]] = {}
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount files for serving previews
app.mount("/static_files", StaticFiles(directory=WORKSPACE_DIR), name="static_files")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(WORKSPACE_DIR, "web", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

def run_translation_task(
    task_id: str,
    uploaded_path: str,
    target_lang: str,
    translate_free_text: bool = False,
    series_id: Optional[str] = None,
    auto_accept_glossary: bool = False,
    translation_backend: str = "auto"
):
    task = TASKS[task_id]
    t0 = time.time()

    try:
        task["status_message"] = "Mengekstrak halaman dokumen..."
        task["percent"] = 5
        task_dir = os.path.join(TEMP_DIR, task_id)
        extract_dir = os.path.join(task_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        # 1. Ekstrak halaman
        pages = DocumentProcessor.extract_pages(uploaded_path, extract_dir)
        total_pages = len(pages)
        task["total_pages"] = total_pages
        task["percent"] = 10

        engine = TranslationEngine.get_instance()

        id_pages = []
        en_pages = []
        orig_web_paths = []
        trans_web_paths = []
        low_quality_bubbles_all = []

        # Translation memory & Glosarium per seri
        series_glossary = engine.glossary_mgr.load_glossary(series_id) if series_id else {}
        trans_memory = ChapterTranslationMemory(initial_glossary=series_glossary)

        # Lock bahasa dokumen (diambil dari halaman pertama agar konsisten & cepat)
        document_language_lock = None

        # 2. Proses tiap halaman dengan BATCH GPU
        for i, page_path in enumerate(pages):
            # Cek cancel
            if task.get("is_cancelled", False):
                print(f"[TASK {task_id}] Dibatalkan oleh pengguna pada halaman {i+1}.")
                task["status_message"] = "Proses dibatalkan oleh pengguna."
                task["detail_message"] = f"Dibatalkan pada halaman {i+1} dari {total_pages}."
                return

            task["status_message"] = f"Menerjemahkan halaman {i+1} dari {total_pages}..."
            task["detail_message"] = f"Memproses paralel batch balon kata halaman #{i+1}"
            progress = 10 + int(((i + 1) / total_pages) * 80)
            task["percent"] = progress

            res = engine.process_single_page(
                page_path,
                target_lang=target_lang,
                doc_lang_hint=document_language_lock,
                save_debug=True,
                debug_dir=os.path.join(task_dir, "debug"),
                translate_free_text=translate_free_text,
                series_id=series_id,
                translation_memory=trans_memory,
                translation_backend=translation_backend
            )

            # Kunci bahasa dokumen dari deteksi pertama
            if document_language_lock is None and res.get("detected_lang"):
                document_language_lock = res["detected_lang"]
                print(f"[OPTIMASI] Bahasa dokumen dikunci ke: {document_language_lock} ({get_language_name(document_language_lock)})")

            if res.get("debug_dump", {}).get("low_quality_bubbles"):
                for lqb in res["debug_dump"]["low_quality_bubbles"]:
                    low_quality_bubbles_all.append({
                        "page": i + 1,
                        "bubble_id": lqb["bubble_id"],
                        "ocr_text": lqb["ocr_text"],
                        "ocr_quality": lqb["ocr_quality"]
                    })

            res_images = res["images"]
            base = os.path.splitext(os.path.basename(page_path))[0]

            if "ind_Latn" in res_images:
                id_path = os.path.join(task_dir, f"{base}_id.png")
                res_images["ind_Latn"].save(id_path)
                id_pages.append(id_path)

            if "eng_Latn" in res_images:
                en_path = os.path.join(task_dir, f"{base}_en.png")
                res_images["eng_Latn"].save(en_path)
                en_pages.append(en_path)

            orig_rel = os.path.relpath(page_path, WORKSPACE_DIR).replace("\\", "/")
            orig_web_paths.append(f"/static_files/{orig_rel}")

            trans_page = id_pages[-1] if id_pages else en_pages[-1]
            trans_rel = os.path.relpath(trans_page, WORKSPACE_DIR).replace("\\", "/")
            trans_web_paths.append(f"/static_files/{trans_rel}")

        # Cek cancel sebelum perakitan
        if task.get("is_cancelled", False):
            task["status_message"] = "Proses dibatalkan oleh pengguna."
            return

        # 3. Rakit output akhir
        task["status_message"] = "Mengemas file dokumen terjemahan..."
        task["percent"] = 95

        base_name = os.path.splitext(os.path.basename(uploaded_path))[0]
        ext = os.path.splitext(uploaded_path)[1].lower()
        downloads = []

        if id_pages:
            out_id_file = os.path.join(OUTPUT_DIR, f"{base_name}_indonesia{ext}")
            DocumentProcessor.assemble_output(uploaded_path, id_pages, out_id_file)
            rel_id = os.path.relpath(out_id_file, WORKSPACE_DIR).replace("\\", "/")
            downloads.append({
                "label": f"Versi Indonesia ({ext.upper()})",
                "filename": os.path.basename(out_id_file),
                "url": f"/static_files/{rel_id}"
            })

        if en_pages:
            out_en_file = os.path.join(OUTPUT_DIR, f"{base_name}_english{ext}")
            DocumentProcessor.assemble_output(uploaded_path, en_pages, out_en_file)
            rel_en = os.path.relpath(out_en_file, WORKSPACE_DIR).replace("\\", "/")
            downloads.append({
                "label": f"Versi Inggris ({ext.upper()})",
                "filename": os.path.basename(out_en_file),
                "url": f"/static_files/{rel_en}"
            })

        # Ekstraksi kandidat nama tokoh/istilah untuk konfirmasi pengguna di UI
        task_debug_dir = os.path.join(task_dir, "debug")
        debug_files = sorted([f for f in os.listdir(task_debug_dir) if f.endswith("_debug.json")]) if os.path.exists(task_debug_dir) else []
        chapter_texts = []
        for df in debug_files:
            try:
                with open(os.path.join(task_debug_dir, df), "r", encoding="utf-8") as jf:
                    pdata = json.load(jf)
                    chapter_texts.append([b.get("ocr_text", "") for b in pdata.get("bubbles", []) if b.get("ocr_text")])
            except Exception:
                pass

        discovered_candidates = engine.name_extractor.extract_candidates(chapter_texts) if chapter_texts else []

        # 4. Selesai
        task["is_completed"] = True
        task["percent"] = 100
        task["status_message"] = "Penerjemahan Selesai!"
        task["detail_message"] = f"Total {total_pages} halaman selesai dalam {round(time.time() - t0, 1)} detik."
        task["result"] = {
            "total_pages": total_pages,
            "duration_seconds": round(time.time() - t0, 1),
            "downloads": downloads,
            "original_pages": orig_web_paths,
            "translated_pages": trans_web_paths,
            "low_quality_bubbles": low_quality_bubbles_all,
            "name_candidates": discovered_candidates
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["is_error"] = True
        task["error_message"] = str(e)

class GlossaryUpdateRequest(BaseModel):
    glossary: Dict[str, str]

class GlossaryAcceptItem(BaseModel):
    term: str
    translation: str

class GlossaryAcceptRequest(BaseModel):
    terms: List[GlossaryAcceptItem]

def _validate_uuid_task_id(task_id: str) -> str:
    """
    Validasi parameter task_id untuk mencegah eksploitasi path traversal.
    Mendukung format standard UUID (36 karakter) dan 8-karakter hex.
    """
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id tidak boleh kosong.")
    clean = task_id.strip()
    if not re.fullmatch(r'^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})?$', clean):
        raise HTTPException(status_code=400, detail="Format task_id harus berupa UUID yang valid.")
    return clean

@app.post("/api/translate")
async def start_translation(
    file: UploadFile = File(...),
    target_lang: str = Form("ind_Latn"),
    translate_free_text: bool = Form(False),
    series_id: Optional[str] = Form(None),
    auto_accept_glossary: bool = Form(False),
    translation_backend: str = Form("auto")
):
    task_id = str(uuid.uuid4())[:8]
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Sanitasi nama file upload untuk mencegah path traversal dan karakter berbahaya
    raw_name = os.path.basename(file.filename or "uploaded_manga")
    safe_filename = re.sub(r'[^a-zA-Z0-9._\-]', '_', raw_name)
    if not safe_filename or safe_filename.startswith('.'):
        safe_filename = f"manga_{safe_filename.lstrip('.')}"
    uploaded_path = os.path.join(task_dir, safe_filename)

    with open(uploaded_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    TASKS[task_id] = {
        "task_id": task_id,
        "filename": safe_filename,
        "series_id": series_id,
        "translation_backend": translation_backend,
        "is_completed": False,
        "is_cancelled": False,
        "is_error": False,
        "percent": 0,
        "status_message": "Memulai proses...",
        "detail_message": ""
    }

    thread = threading.Thread(
        target=run_translation_task,
        args=(task_id, uploaded_path, target_lang, translate_free_text, series_id, auto_accept_glossary, translation_backend),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id, "filename": safe_filename}

@app.get("/api/glossary/{series_id}")
async def get_series_glossary(series_id: str):
    engine = TranslationEngine.get_instance()
    glossary = engine.glossary_mgr.load_glossary(series_id)
    return {"series_id": series_id, "glossary": glossary}

@app.post("/api/glossary/{series_id}")
async def update_series_glossary(series_id: str, req: GlossaryUpdateRequest):
    engine = TranslationEngine.get_instance()
    saved_path = engine.glossary_mgr.save_glossary(series_id, req.glossary)
    return {"status": "ok", "series_id": series_id, "glossary": req.glossary, "path": saved_path}

@app.post("/api/glossary/{series_id}/accept")
async def accept_candidates(series_id: str, req: GlossaryAcceptRequest):
    engine = TranslationEngine.get_instance()
    glossary = engine.glossary_mgr.load_glossary(series_id)
    for item in req.terms:
        term = item.term.strip()
        trans = item.translation.strip()
        if term and trans:
            glossary[term] = trans
    engine.glossary_mgr.save_glossary(series_id, glossary)
    return {"status": "ok", "series_id": series_id, "glossary": glossary}

ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "127.0.0.1:7860", "localhost:7860"}

@app.get("/api/glossary/{series_id}/discover/{task_id}")
async def discover_candidates_from_task(series_id: str, task_id: str):
    valid_task_id = _validate_uuid_task_id(task_id)
    engine = TranslationEngine.get_instance()
    task_dir = os.path.join(TEMP_DIR, valid_task_id, "debug")
    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail="Debug task belum tersedia.")
    files = sorted([f for f in os.listdir(task_dir) if f.endswith("_debug.json")])
    chapter_texts = []
    for f in files:
        with open(os.path.join(task_dir, f), "r", encoding="utf-8") as jf:
            data = json.load(jf)
            page_lines = [b.get("ocr_text", "") for b in data.get("bubbles", []) if b.get("ocr_text")]
            chapter_texts.append(page_lines)

    candidates = engine.name_extractor.extract_candidates(chapter_texts)
    return {
        "series_id": series_id,
        "total_candidates": len(candidates),
        "candidates": candidates
    }

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    valid_task_id = _validate_uuid_task_id(task_id)
    if valid_task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task tidak ditemukan.")
    TASKS[valid_task_id]["is_cancelled"] = True
    TASKS[valid_task_id]["status_message"] = "Membatalkan proses..."
    return {"status": "cancelling", "task_id": valid_task_id}

@app.post("/api/shutdown")
async def shutdown_server(request: Request):
    """
    Mematikan server lokal, melepaskan seluruh VRAM GPU, dan menutup proses Python.
    Hanya menerima permintaan dari Origin / Host lokal terpercaya untuk mencegah serangan CSRF.
    """
    host = (request.headers.get("host") or "").lower()
    origin = (request.headers.get("origin") or "").lower()
    referer = (request.headers.get("referer") or "").lower()

    if host and host not in ALLOWED_LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Akses ditolak: Host tidak valid")

    if origin and not (origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost")):
        raise HTTPException(status_code=403, detail="Akses ditolak: Origin tidak diizinkan")

    if referer and not (referer.startswith("http://127.0.0.1") or referer.startswith("http://localhost")):
        raise HTTPException(status_code=403, detail="Akses ditolak: Referer tidak diizinkan")

    def delayed_exit():
        time.sleep(0.6)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return {"status": "shutting_down", "message": "Server sedang dimatikan."}

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    valid_task_id = _validate_uuid_task_id(task_id)
    if valid_task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task tidak ditemukan.")
    return TASKS[valid_task_id]

@app.get("/api/task/{task_id}/debug/{page_idx}")
async def get_task_debug(task_id: str, page_idx: int):
    valid_task_id = _validate_uuid_task_id(task_id)
    task_dir = os.path.join(TEMP_DIR, valid_task_id)
    debug_dir = os.path.join(task_dir, "debug")
    if not os.path.exists(debug_dir):
        raise HTTPException(status_code=404, detail="Debug data belum tersedia.")
    files = sorted([f for f in os.listdir(debug_dir) if f.endswith("_debug.json")])
    if page_idx < 1 or page_idx > len(files):
        raise HTTPException(status_code=404, detail="Halaman debug tidak ditemukan.")
    file_path = os.path.join(debug_dir, files[page_idx - 1])
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    TranslationEngine.get_instance()
    print("\n" + "=" * 65)
    print("🚀 Manga Translator Fast Engine Aktif!")
    print("👉 Buka di browser: http://127.0.0.1:7860")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
