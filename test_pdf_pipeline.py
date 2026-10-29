import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.document_processor import DocumentProcessor
from app import TranslationEngine

def test_pdf():
    pdf_in = "input/sample_manga_chapter.pdf"
    if not os.path.exists(pdf_in):
        from create_sample_pdf import make_sample_pdf
        make_sample_pdf()

    print(f"Menguji translasi dokumen PDF: {pdf_in}")
    temp_dir = "temp/test_pdf_extract"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Ekstrak halaman PDF
    pages = DocumentProcessor.extract_pages(pdf_in, temp_dir)
    print(f"Berhasil mengekstrak {len(pages)} halaman dari PDF:")
    for p in pages:
        print(f" -> {p}")

    # 2. Terjemahkan tiap halaman
    engine = TranslationEngine.get_instance()
    translated_pages = []

    for i, p in enumerate(pages):
        print(f"Menerjemahkan halaman #{i+1}...")
        res = engine.process_single_image(p, target_lang="ind_Latn")
        out_page = os.path.join(temp_dir, f"trans_page_{i+1}.png")
        res["ind_Latn"].save(out_page)
        translated_pages.append(out_page)

    # 3. Rakit kembali ke format PDF
    pdf_out = "output/sample_manga_chapter_indonesia.pdf"
    DocumentProcessor.assemble_output(pdf_in, translated_pages, pdf_out)
    print(f"\n[SUKSES] Dokumen PDF Terjemahan berhasil dibuat di:\n -> {pdf_out}")
    print(f"Ukuran file: {os.path.getsize(pdf_out)} bytes")

if __name__ == "__main__":
    test_pdf()
