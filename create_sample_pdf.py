import os
import fitz
from PIL import Image

def make_sample_pdf():
    os.makedirs("input", exist_ok=True)
    img_path = "input/sample_manga_page.png"
    if not os.path.exists(img_path):
        from create_sample_manga import generate_sample_manga_page
        generate_sample_manga_page(img_path)

    # Buat PDF 2 halaman dari gambar sampel
    pdf_path = "input/sample_manga_chapter.pdf"
    doc = fitz.open()
    for _ in range(2):
        img_doc = fitz.open(img_path)
        pdf_bytes = img_doc.convert_to_pdf()
        page_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(page_pdf)
        page_pdf.close()
        img_doc.close()

    doc.save(pdf_path)
    doc.close()
    print(f"Sample PDF Manga 2-halaman berhasil dibuat di: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    make_sample_pdf()
