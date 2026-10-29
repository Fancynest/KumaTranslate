import re
from typing import List
from langdetect import detect, DetectorFactory

# Pastikan hasil konsisten
DetectorFactory.seed = 0

LANGUAGE_NAMES = {
    "jpn_Jpan": "Jepang (Manga)",
    "kor_Hang": "Korea (Manhwa)",
    "zho_Hans": "Mandarin (Manhua)",
    "eng_Latn": "Inggris",
    "ind_Latn": "Indonesia",
    "spa_Latn": "Spanyol",
    "fra_Latn": "Prancis",
    "deu_Latn": "Jerman"
}

def detect_language(text: str) -> str:
    """
    Otomatis mendeteksi bahasa teks komik:
    1. Jika ada Hiragana / Katakana -> 100% Jepang (Manga)
    2. Jika ada Hangul -> 100% Korea (Manhwa)
    3. Jika teks hanya Kanji/Hanzi:
       - Cek partikel umum bahasa Mandarin (的, 是, 了, 我, 们, 个, 这, 不, 说)
       - Jika tidak ada partikel Mandarin, default ke Jepang (karena manga penuh kanji seperti '係船柱', '先輩', '先生')
    4. Teks alfabet Latin -> Deteksi Inggris/Eropa
    """
    if not text or not text.strip():
        return "jpn_Jpan"

    clean_text = text.strip()

    # 1. Cek Karakter Hiragana / Katakana (Jepang - Manga)
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', clean_text):
        return "jpn_Jpan"

    # 2. Cek Karakter Hangul (Korea - Manhwa)
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', clean_text):
        return "kor_Hang"

    # 3. Cek Karakter Hanzi / Kanji
    if re.search(r'[\u4e00-\u9fff]', clean_text):
        # Cek penanda tata bahasa khas Mandarin (Simplified/Traditional)
        if re.search(r'[的是了我他在们这一个个不要说很]', clean_text):
            return "zho_Hans"
        # Jika kanji murni tanpa partikel mandarin (misal kanji nama jurus / panggilan),
        # di komik manga jepang ini adalah Bahasa Jepang!
        return "jpn_Jpan"

    # 4. Teks Alfabet Latin
    try:
        iso_code = detect(clean_text)
        mapping = {
            "en": "eng_Latn",
            "id": "ind_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn"
        }
        return mapping.get(iso_code, "eng_Latn")
    except Exception:
        return "eng_Latn"

def detect_document_language(texts: List[str]) -> str:
    """
    Mendeteksi bahasa dokumen secara holistik dari kumpulan seluruh balon kata:
    - Jika ada minimal SATU balon mengandung Kana (Hiragana/Katakana) -> Dokumen adalah Jepang
    - Jika ada minimal SATU balon mengandung Hangul -> Dokumen adalah Korea
    - Jika dominan Hanzi tanpa Kana/Hangul -> Mandarin
    - Lainnya -> Latin/Inggris
    """
    if not texts:
        return "jpn_Jpan"

    combined = " ".join([t for t in texts if t])
    if not combined:
        return "jpn_Jpan"

    # Kana hadir -> 100% Manga Jepang
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', combined):
        return "jpn_Jpan"

    # Hangul hadir -> 100% Manhwa Korea
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', combined):
        return "kor_Hang"

    # Hanzi hadir
    if re.search(r'[\u4e00-\u9fff]', combined):
        if re.search(r'[的是了我他在们这一个个不要说很]', combined):
            return "zho_Hans"
        return "jpn_Jpan"

    # Latin
    try:
        iso = detect(combined)
        mapping = {"en": "eng_Latn", "id": "ind_Latn"}
        return mapping.get(iso, "eng_Latn")
    except Exception:
        return "eng_Latn"

def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)

if __name__ == "__main__":
    tests = [
        "お前はもう死んでいる",
        "係船柱！",
        "안녕하세요!",
        "我是一名海贼！",
        "Hello there!"
    ]
    for t in tests:
        print(f"{t} -> {detect_language(t)} ({get_language_name(detect_language(t))})")
