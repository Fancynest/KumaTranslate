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
    1. Jika teks alfabet Latin memiliki kata kamus bahasa Inggris valid -> Inggris
    2. Jika ada Hiragana / Katakana signifikan -> Jepang (Manga)
    3. Jika ada Hangul -> Korea (Manhwa)
    4. Jika teks hanya Kanji/Hanzi:
       - Cek partikel umum bahasa Mandarin (的是了我他在们这一个个不要说很)
       - Jika tidak ada partikel Mandarin, default ke Jepang
    5. Teks alfabet Latin -> Deteksi Inggris/Eropa
    """
    if not text or not text.strip():
        return "jpn_Jpan"

    clean_text = text.strip()

    from src.text_utils import count_english_words
    en_words = count_english_words(clean_text)
    cjk_chars = len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', clean_text))
    hangul_chars = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', clean_text))
    latin_chars = len(re.findall(r'[A-Za-z]', clean_text))

    # 1. Jika terbukti kata kamus bahasa Inggris dominan, maka adalah Latin / English
    if en_words >= 2 or (en_words >= 1 and latin_chars > cjk_chars * 2):
        return "eng_Latn"

    # 2. Cek Karakter Hangul (Korea - Manhwa)
    if hangul_chars >= 2 or (hangul_chars > 0 and hangul_chars >= cjk_chars):
        return "kor_Hang"

    # 3. Cek Karakter Hiragana / Katakana (Jepang - Manga)
    kana_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', clean_text))
    if kana_chars >= 2 or (kana_chars > 0 and cjk_chars >= latin_chars):
        return "jpn_Jpan"

    # 4. Cek Karakter Hanzi / Kanji
    if cjk_chars > 0 and cjk_chars >= latin_chars:
        # Cek penanda tata bahasa khas Mandarin (Simplified/Traditional)
        if re.search(r'[的是了我他在们这一个个不要说很]', clean_text):
            return "zho_Hans"
        return "jpn_Jpan"

    # 5. Teks Alfabet Latin
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
    - Menghitung statistik kata kamus bahasa Inggris, karakter CJK (Jepang), dan Hangul (Korea).
    - Memilih bahasa yang mendominasi dokumen, mencegah 1 karakter halusinasi merusak seluruh halaman.
    """
    if not texts:
        return "jpn_Jpan"

    valid_texts = [t.strip() for t in texts if t and t.strip()]
    if not valid_texts:
        return "jpn_Jpan"

    combined = " ".join(valid_texts)

    from src.text_utils import count_english_words
    total_en_words = sum(count_english_words(t) for t in valid_texts)
    total_latin_chars = sum(len(re.findall(r'[A-Za-z]', t)) for t in valid_texts)
    total_kana_chars = sum(len(re.findall(r'[\u3040-\u30ff]', t)) for t in valid_texts)
    total_cjk_chars = sum(len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', t)) for t in valid_texts)
    total_hangul_chars = sum(len(re.findall(r'[\uac00-\ud7af]', t)) for t in valid_texts)

    # 1. Jika dokumen didominasi kata kamus bahasa Inggris valid
    if total_en_words >= 3 and total_en_words >= total_cjk_chars:
        return "eng_Latn"
    if total_latin_chars > 20 and total_latin_chars > total_cjk_chars * 3:
        return "eng_Latn"

    # 2. Jika dokumen didominasi karakter Hangul (Korea - Manhwa)
    if total_hangul_chars >= 3 and total_hangul_chars >= total_cjk_chars:
        return "kor_Hang"

    # 3. Jika dokumen mengandung Kana Jepang yang jelas (>= 2 karakter Kana di dokumen)
    if total_kana_chars >= 2:
        return "jpn_Jpan"

    # 4. Jika didominasi Hanzi tanpa Kana
    if total_cjk_chars > 0 and total_cjk_chars >= total_latin_chars:
        if re.search(r'[的是了我他在们这一个个不要说很]', combined):
            return "zho_Hans"
        return "jpn_Jpan"

    # 5. Fallback ke deteksi teks Latin
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
        "THIS IS ONE OF THE ISLANDS IN THE SETO INLAND SEA.",
        "手土産を持ってきてやったぞ",
        "진우는 어디로 갔는가?",
        "这是我的漫画作品"
    ]
    for t in tests:
        print(f"{t} -> {detect_language(t)} ({get_language_name(detect_language(t))})")
