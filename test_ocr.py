import sys
import os

# Pastikan console Windows mendukung output UTF-8 untuk karakter Jepang
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFont
from src.ocr import MangaOCRService

def create_sample_bubble(text="お前はもう死んでいる", output_path="input/sample_japanese.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Buat kanvas putih menyerupai balon kata
    width, height = 300, 150
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Gambar border balon kata
    draw.rounded_rectangle([10, 10, width - 10, height - 10], radius=20, outline=(0, 0, 0), width=4)

    # Gunakan font Jepang bawaan Windows (MS Gothic)
    font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 26)
    else:
        font = ImageFont.load_default()

    # Hitung posisi teks agar pas di tengah balon
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((width - text_w) // 2, (height - text_h) // 2)

    draw.text(pos, text, fill=(0, 0, 0), font=font)
    img.save(output_path)
    print(f"Sample gambar manga dibuat di: {output_path}")
    print(f"Teks asli yang ditulis ke gambar: {text}")
    return output_path

if __name__ == "__main__":
    sample_file = create_sample_bubble()
    
    ocr = MangaOCRService()
    hasil_ocr = ocr.read_text(sample_file)
    print("=" * 45)
    print(f"[HASIL OCR MANGA-OCR]: {hasil_ocr}")
    print("=" * 45)
