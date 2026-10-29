import os
import sys
import uuid
import time
import shutil
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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
        print("[ENGINE] Seluruh model AI berhasil dimuat.")

    def process_single_page(
        self,
        img_path: str,
        target_lang: str = "ind_Latn",
        doc_lang_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Memproses 1 halaman secara BATCH PARALEL Presisi Tinggi:
        - Deteksi balon kata (Normalisasi skala cerdas + Filter banner judul)
        - OCR dengan Manga-OCR / RapidOCR
        - Batch translation ke GPU NLLB-200 FP16
        - Stroke-level inpainting (0% merusak gambar karakter)
        - Strict Fitting Typesetting (100% DIJAMIN teks TIDAK KELUAR dari balon kata)
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
            return {
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "images": {"ind_Latn": img, "eng_Latn": img}
            }

        # 1. OCR teks dari tiap balon kata
        raw_texts = []
        valid_bubbles = []

        for b in bubbles:
            raw = self.ocr.read_text(b["crop"], lang_hint=doc_lang_hint)
            if raw and raw.strip():
                t_clean = raw.strip()
                # Abaikan bercak tanda baca tunggal
                if len(t_clean) == 1 and t_clean in ".,-~:;!・":
                    continue
                raw_texts.append(t_clean)
                valid_bubbles.append(b)

        if not raw_texts:
            return {
                "detected_lang": doc_lang_hint or "jpn_Jpan",
                "images": {"ind_Latn": img, "eng_Latn": img}
            }

        # 2. Tentukan bahasa dokumen secara holistik
        if doc_lang_hint:
            page_lang = doc_lang_hint
        else:
            page_lang = detect_document_language(raw_texts)

        # 3. BATCH TRANSLATE (Semua balon sekaligus dalam 1 panggilan GPU)
        res_id_texts = []
        res_en_texts = []

        if target_lang in ["ind_Latn", "both"]:
            res_id_texts = self.translator.translate_batch(raw_texts, src_lang=page_lang, tgt_lang="ind_Latn")

        if target_lang in ["eng_Latn", "both"]:
            res_en_texts = self.translator.translate_batch(raw_texts, src_lang=page_lang, tgt_lang="eng_Latn")

        # 4. Inpainting: Bersihkan teks asli
        #    - text_bubble: bersihkan goresan teks di seluruh rongga balon kata (bubble_bbox + bubble_mask)
        #      sehingga tidak ada goresan kana/kanji yang tertinggal meskipun ada baris yang terlewat.
        #    - free_text: bersihkan hanya di area text_bbox agar tidak merusak artwork sekitar.
        inpaint_boxes = []
        inpaint_masks = []
        for b in valid_bubbles:
            if b.get("type") == "text_bubble":
                inpaint_boxes.append(b.get("bubble_bbox", b["bbox"]))
                inpaint_masks.append(b.get("bubble_mask"))
            else:
                inpaint_boxes.append(b.get("text_bbox", b["bbox"]))
                inpaint_masks.append(b.get("bubble_mask"))

        cleaned_img = self.inpainter.clean_bubbles(img, inpaint_boxes, masks=inpaint_masks)

        # 5. Typesetting: Render teks terjemahan ke dalam safe_bbox (rongga dalam balon kata)
        #    Teks terjemahan horizontal (Indonesia / Inggris) membutuhkan ruang lebar balon kata.
        #    Renderer strict fitting menjamin teks 100% terkunci rapi di dalam rongga tanpa meluber keluar.
        output_images = {}
        if target_lang in ["ind_Latn", "both"]:
            img_id = cleaned_img.copy()
            for b, t in zip(valid_bubbles, res_id_texts):
                render_box = b.get("safe_bbox", b.get("bubble_bbox", b["bbox"]))
                img_id = self.renderer.render_bubble_text(img_id, render_box, t)
            output_images["ind_Latn"] = img_id

        if target_lang in ["eng_Latn", "both"]:
            img_en = cleaned_img.copy()
            for b, t in zip(valid_bubbles, res_en_texts):
                render_box = b.get("safe_bbox", b.get("bubble_bbox", b["bbox"]))
                img_en = self.renderer.render_bubble_text(img_en, render_box, t)
            output_images["eng_Latn"] = img_en

        return {
            "detected_lang": page_lang,
            "images": output_images
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

def run_translation_task(task_id: str, uploaded_path: str, target_lang: str):
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
                doc_lang_hint=document_language_lock
            )

            # Kunci bahasa dokumen dari deteksi pertama
            if document_language_lock is None and res.get("detected_lang"):
                document_language_lock = res["detected_lang"]
                print(f"[OPTIMASI] Bahasa dokumen dikunci ke: {document_language_lock} ({get_language_name(document_language_lock)})")

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
            "translated_pages": trans_web_paths
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["is_error"] = True
        task["error_message"] = str(e)

@app.post("/api/translate")
async def start_translation(
    file: UploadFile = File(...),
    target_lang: str = Form("ind_Latn")
):
    task_id = str(uuid.uuid4())[:8]
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    uploaded_path = os.path.join(task_dir, file.filename)
    with open(uploaded_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    TASKS[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "is_completed": False,
        "is_cancelled": False,
        "is_error": False,
        "percent": 0,
        "status_message": "Memulai proses...",
        "detail_message": ""
    }

    thread = threading.Thread(
        target=run_translation_task,
        args=(task_id, uploaded_path, target_lang),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id, "filename": file.filename}

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task tidak ditemukan.")
    TASKS[task_id]["is_cancelled"] = True
    TASKS[task_id]["status_message"] = "Membatalkan proses..."
    return {"status": "cancelling", "task_id": task_id}

@app.post("/api/shutdown")
async def shutdown_server():
    """
    Mematikan server lokal, melepaskan seluruh VRAM GPU, dan menutup proses Python.
    """
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
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task tidak ditemukan.")
    return TASKS[task_id]

if __name__ == "__main__":
    import uvicorn
    TranslationEngine.get_instance()
    print("\n" + "=" * 65)
    print("🚀 Manga Translator Fast Engine Aktif!")
    print("👉 Buka di browser: http://127.0.0.1:7860")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
