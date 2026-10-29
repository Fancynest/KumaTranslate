import os
import pymupdf as fitz
import zipfile
import shutil
from PIL import Image
from typing import List, Tuple, Optional

class DocumentProcessor:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ARCHIVE_EXTS = {".zip", ".cbz"}
    PDF_EXTS = {".pdf"}

    @classmethod
    def get_file_type(cls, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in cls.IMAGE_EXTS:
            return "image"
        elif ext in cls.PDF_EXTS:
            return "pdf"
        elif ext in cls.ARCHIVE_EXTS:
            return "archive"
        else:
            return "unknown"

    @classmethod
    def extract_pages(cls, file_path: str, extract_dir: str) -> List[str]:
        """
        Mengekstrak seluruh halaman dari file (Image, PDF, ZIP, CBZ).
        Mengembalikan list path file gambar sementara secara berurutan.
        """
        os.makedirs(extract_dir, exist_ok=True)
        file_type = cls.get_file_type(file_path)

        extracted_paths = []

        if file_type == "image":
            # Salin gambar tunggal ke folder kerja
            dest = os.path.join(extract_dir, f"page_001{os.path.splitext(file_path)[1]}")
            shutil.copy2(file_path, dest)
            extracted_paths.append(dest)

        elif file_type == "pdf":
            # Ekstrak tiap halaman PDF menjadi gambar PNG tajam (DPI 150)
            doc = fitz.open(file_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=150)
                page_filename = f"page_{page_idx+1:04d}.png"
                out_path = os.path.join(extract_dir, page_filename)
                pix.save(out_path)
                extracted_paths.append(out_path)
            doc.close()

        elif file_type == "archive":
            # Ekstrak file ZIP / CBZ
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Ambil hanya file gambar dan urutkan namanya
                namelist = sorted([n for n in zf.namelist() if os.path.splitext(n)[1].lower() in cls.IMAGE_EXTS])
                for idx, name in enumerate(namelist):
                    ext = os.path.splitext(name)[1].lower()
                    target_filename = f"page_{idx+1:04d}{ext}"
                    target_path = os.path.join(extract_dir, target_filename)
                    with zf.open(name) as src_file, open(target_path, 'wb') as dst_file:
                        shutil.copyfileobj(src_file, dst_file)
                    extracted_paths.append(target_path)

        else:
            raise ValueError(f"Format file tidak didukung: {file_path}")

        return extracted_paths

    @classmethod
    def assemble_output(cls, original_file_path: str, translated_image_paths: List[str], output_path: str) -> str:
        """
        Mengemas kembali hasil terjemahan sesuai format file aslinya:
        - Input PDF  -> Output PDF
        - Input ZIP/CBZ -> Output ZIP/CBZ
        - Input Gambar -> Output Gambar
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        file_type = cls.get_file_type(original_file_path)
        ext = os.path.splitext(original_file_path)[1].lower()

        if file_type == "pdf":
            # Gabungkan gambar-gambar terjemahan menjadi dokumen PDF berkualitas tinggi
            pdf_doc = fitz.open()
            for img_path in translated_image_paths:
                img_doc = fitz.open(img_path)
                pdf_bytes = img_doc.convert_to_pdf()
                img_pdf = fitz.open("pdf", pdf_bytes)
                pdf_doc.insert_pdf(img_pdf)
                img_pdf.close()
                img_doc.close()
            
            pdf_doc.save(output_path)
            pdf_doc.close()
            return output_path

        elif file_type == "archive":
            # Kemas kembali ke dalam arsip ZIP / CBZ
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for img_path in translated_image_paths:
                    zf.write(img_path, os.path.basename(img_path))
            return output_path

        elif file_type == "image":
            # Salin gambar terjemahan tunggal
            shutil.copy2(translated_image_paths[0], output_path)
            return output_path

        else:
            raise ValueError(f"Format tidak dikenali: {original_file_path}")

if __name__ == "__main__":
    print("DocumentProcessor Module siap digunakan.")
