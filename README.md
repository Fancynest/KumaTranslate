# KumaTranslate 🐻

**KumaTranslate** adalah aplikasi penerjemah Manga, Manhwa, Manhua, dan Komik digital berbasis AI lokal yang berjalan **100% offline** di komputer kamu dengan akselerasi GPU (CUDA FP16).

---

## ✨ Fitur Unggulan

- **Dukungan Format Lengkap**:
  - Dokumen PDF manga (*input PDF $\rightarrow$ output terjemahan kembali menjadi format PDF*)
  - Arsip komik (ZIP, CBZ)
  - Gambar langsung (PNG, JPG, WEBP)
- **Auto-Detect Bahasa Sumber Cerdas**:
  - 🇯🇵 **Manga (Jepang)**: Vision Transformer (*Manga-OCR*)
  - 🇰🇷 **Manhwa (Korea)**: Deep Learning OCR (*EasyOCR*)
  - 🇨🇳 **Manhua (Mandarin)**: High-speed ONNX OCR (*RapidOCR*)
  - 🇺🇸 **Comics / Webtoons (Inggris / Barat)**: (*RapidOCR*)
- **Bahasa Tujuan Fleksibel**:
  - Bahasa Indonesia (`ind_Latn`)
  - Bahasa Inggris (`eng_Latn`)
  - Atau keduanya sekaligus (*Dual Output*)
- **Preservasi Artwork & Anti-Bocor**:
  - **Title Banner Protection**: Judul chapter, credit pengarang, dan cover artwork tidak dirusak.
  - **Stroke-Level Inpainting**: Hanya menghapus goresan tinta teks asli tanpa merusak gambar karakter atau membuat kotak putih.
  - **Strict Inner Bubble Fitting**: Teks terjemahan otomatis disesuaikan ukurannya dan terkunci 100% di dalam rongga balon kata (tidak keluar dari balon).
- **Modern Web UI & 1-Click Launcher**:
  - Antarmuka web modern dark-mode dengan Inter font, drag & drop, live progress bar, pembatalan instan, dan tombol shutdown (untuk membebaskan RAM/VRAM saat selesai).
  - Shortcut 1-klik di Desktop Windows.

---

## 🛠️ Instalasi & Persyaratan

### Persyaratan Sistem:
- Python 3.10 / 3.11
- GPU NVIDIA dengan dukungan CUDA (disarankan minimal 4GB VRAM, misal RTX 2050 / GTX 1650 ke atas) atau CPU mode.

### Instalasi Dependensi:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Web UI (Rekomendasi)
Cukup jalankan:
```bash
python app.py
```
atau klik dua kali file **`run_app.bat`** (atau shortcut Desktop **`Manga Translator`**).
Buka di browser:
👉 **http://127.0.0.1:7860**

### 2. Menjalankan via CLI
```bash
python main.py --image input/sample_manga_page.png --lang ind_Latn
```

---

## 📂 Struktur Proyek

```text
├── src/
│   ├── detector.py           # DBNet bubble detector & safe contour scanner
│   ├── ocr.py                # Multi-language OCR engine (Manga-OCR, RapidOCR, EasyOCR)
│   ├── translator.py         # NLLB-200 FP16 GPU batch translator
│   ├── inpainter.py          # Stroke-level Navier-Stokes inpainter
│   ├── renderer.py           # Strict auto-fit comic typesetter
│   ├── lang_detector.py      # Holistic comic script & language router
│   └── document_processor.py # PDF, CBZ, ZIP, & Image processor (PyMuPDF)
├── web/
│   └── index.html            # Dark-mode Web UI
├── app.py                    # FastAPI server & background task worker
├── requirements.txt          # Python dependencies
├── run_app.bat               # Windows batch launcher
├── launcher.ps1              # Desktop shortcut script
└── launch.vbs                # Silent background launcher
```

---

## 📄 Lisensi
MIT License
