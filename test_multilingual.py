import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFont
from src.ocr import MultiLingualOCR
from src.lang_detector import detect_language, get_language_name
from src.translator import MangaTranslator

def create_bubble_image(text: str, font_path: str, font_size: int, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 400, 120
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([5, 5, w - 5, h - 5], radius=15, outline=(0, 0, 0), width=3)

    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((w - text_w) // 2, (h - text_h) // 2)
    draw.text(pos, text, fill=(0, 0, 0), font=font)
    img.save(output_path)
    return output_path

def run_tests():
    ocr = MultiLingualOCR()
    translator = MangaTranslator()

    test_cases = [
        {
            "name": "Manga Jepang",
            "text": "お前はもう死んでいる",
            "font": "C:\\Windows\\Fonts\\msgothic.ttc",
            "size": 24,
            "path": "input/test_jp.png"
        },
        {
            "name": "Manhwa Korea",
            "text": "안녕하세요! 반갑습니다.",
            "font": "C:\\Windows\\Fonts\\malgun.ttf",
            "size": 24,
            "path": "input/test_ko.png"
        },
        {
            "name": "Manhua Mandarin",
            "text": "我是海贼王！",
            "font": "C:\\Windows\\Fonts\\msyh.ttc",
            "size": 24,
            "path": "input/test_zh.png"
        },
        {
            "name": "Comic Barat / Inggris",
            "text": "I will be the Pirate King!",
            "font": "C:\\Windows\\Fonts\\comicbd.ttf",
            "size": 22,
            "path": "input/test_en.png"
        }
    ]

    print("\n" + "=" * 65)
    print("PENGUJIAN END-TO-END: AUTO-DETECT BAHASA & MULTI-TRANSLATE")
    print("=" * 65)

    for tc in test_cases:
        img_path = create_bubble_image(tc["text"], tc["font"], tc["size"], tc["path"])
        
        # 1. OCR (Auto-detect script)
        read_text = ocr.read_text(img_path)

        # 2. Deteksi Bahasa
        src_code = detect_language(read_text)
        src_name = get_language_name(src_code)

        # 3. Terjemah ke Indonesia & Inggris
        trans_id = translator.translate(read_text, src_lang=src_code, tgt_lang="ind_Latn")
        trans_en = translator.translate(read_text, src_lang=src_code, tgt_lang="eng_Latn")

        print(f"Kategori    : {tc['name']}")
        print(f"Teks Asli   : {tc['text']}")
        print(f"Hasil OCR   : {read_text}")
        print(f"Auto-Detect : {src_name} ({src_code})")
        print(f"-> INDONESIA: {trans_id}")
        print(f"-> INGGRIS  : {trans_en}")
        print("-" * 65)

if __name__ == "__main__":
    run_tests()
