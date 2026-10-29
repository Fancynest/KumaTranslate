import re
from typing import List, Dict, Set, Any, Optional

# Kumpulan karakter pemisah / tanda baca & angka untuk mengekstrak teks inti (core)
PUNCT_AND_DIGITS_REGEX = re.compile(
    r'[\s\d\u3000\uFF10-\uFF19、。！？…・「」『』（）()．，!?.,:;~〜\-_"\'`“”‘’\^#@&*+=/\\|<>{}\[\]]'
)

# Pola karakter CJK (Hiragana, Katakana, Kanji/Han, Hangul)
CJK_REGEX = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]')

# Karakter Kana & Han (tanpa Hangul, karena bahasa Korea memakai spasi antar kata)
KANA_HAN_CHAR = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uFF66-\uFF9F]'

# Karakter Katakana termasuk perpanjangan vokal (ー) dan halfwidth
KATAKANA_CHAR = r'[\u30A0-\u30FF\uFF66-\uFF9Fー]'

# Alasan kuat kandidat nama diri tokoh komik
STRONG_NAME_REASONS: Set[str] = {"proper_noun", "before_honorific", "introduced_character"}

# Tanda penutup kutip dan kurung
CLOSING_QUOTES = set("」』）)”\"'’】]>")


def normalize_ocr_text(text: str) -> str:
    """
    Membersihkan teks hasil OCR komik:
    - Hanya memproses jika mengandung CJK (teks non-CJK hanya di-strip).
    - Menghapus spasi/tab/U+3000 di antara dua karakter Kana/Han (JANGAN Hangul).
    - Menghapus garis vertikal sisa OCR (| ｜ 丨 │ ┃).
    - Mengubah elipsis berulang ('・・'/'･･' 2+, '...' 3+, '……', '‥') menjadi '…'.
    - Mempertahankan '・' tunggal untuk nama asing (misal: アルバート・ルイス).
    """
    if not text:
        return ""

    # Teks non-CJK hanya di-strip
    if not CJK_REGEX.search(text):
        return text.strip()

    cleaned = text.strip()

    # 1. Hapus garis vertikal sisa OCR
    cleaned = re.sub(r'[|｜丨│┃]', '', cleaned)

    # 2. Normalisasi variasi elipsis & titik tengah ganda
    # '・・' atau '･･' (2 atau lebih) -> '…' (pertahankan '・' tunggal)
    cleaned = re.sub(r'[・･]{2,}', '…', cleaned)
    # '...' (3 atau lebih titik ASCII) -> '…'
    cleaned = re.sub(r'\.{3,}', '…', cleaned)
    # '……' (2 atau lebih) -> '…'
    cleaned = re.sub(r'…{2,}', '…', cleaned)
    # '‥' (two-dot leader) -> '…'
    cleaned = re.sub(r'‥+', '…', cleaned)

    # 3. Hapus spasi/tab/fullwidth space di antara dua karakter Kana/Han
    # Lookbehind dan lookahead memastikan spasi hanya dihapus jika diapit Kana/Han
    cleaned = re.sub(
        rf'(?<={KANA_HAN_CHAR})[\s\u3000\t]+(?={KANA_HAN_CHAR})',
        '',
        cleaned
    )

    return cleaned.strip()


def split_sentences(text: str) -> List[str]:
    """
    Memecah teks balon kata multi-kalimat setelah tanda baca 。！？!?.
    Aturan:
    - JANGAN memecah di tengah rangkaian tanda baca (misal '！？').
    - JANGAN memecah sebelum tanda penutup kutip/kurung (」』）)”"’】).
    - Pecahan dengan < 2 karakter bermakna ditempel ke segmen tetangganya
      (tanpa spasi jika mengandung CJK).
    """
    if not text:
        return []

    norm_text = text.strip()
    if not norm_text:
        return []

    terminators = set("。！？!?")
    n = len(norm_text)
    split_indices = []

    for i in range(n):
        char = norm_text[i]
        if char in terminators:
            # Periksa apakah di akhir teks
            if i + 1 >= n:
                continue
            next_char = norm_text[i + 1]
            # Jangan pecah di tengah rangkaian tanda baca
            if next_char in terminators:
                continue
            # Jangan pecah sebelum penutup kutip
            if next_char in CLOSING_QUOTES:
                continue
            split_indices.append(i + 1)

    if not split_indices:
        raw_pieces = [norm_text]
    else:
        raw_pieces = []
        prev = 0
        for idx in split_indices:
            part = norm_text[prev:idx].strip()
            if part:
                raw_pieces.append(part)
            prev = idx
        last_part = norm_text[prev:].strip()
        if last_part:
            raw_pieces.append(last_part)

    # Validasi pecahan dengan < 2 karakter bermakna
    merged: List[str] = []
    for piece in raw_pieces:
        core = PUNCT_AND_DIGITS_REGEX.sub('', piece)
        if len(core) < 2 and merged:
            # Tempel ke tetangga sebelumnya
            prev_piece = merged[-1]
            has_cjk = bool(CJK_REGEX.search(prev_piece) or CJK_REGEX.search(piece))
            sep = "" if has_cjk else " "
            merged[-1] = f"{prev_piece}{sep}{piece}"
        else:
            merged.append(piece)

    # Jika elemen pertama memiliki < 2 karakter bermakna dan ada elemen kedua, tempel ke kanan
    if len(merged) >= 2:
        first_core = PUNCT_AND_DIGITS_REGEX.sub('', merged[0])
        if len(first_core) < 2:
            p0 = merged.pop(0)
            has_cjk = bool(CJK_REGEX.search(p0) or CJK_REGEX.search(merged[0]))
            sep = "" if has_cjk else " "
            merged[0] = f"{p0}{sep}{merged[0]}"

    return merged if merged else [norm_text]


def apply_glossary_source(text: str, glossary: Dict[str, str]) -> str:
    """
    Menerapkan pra-substitusi glosarium pada teks sumber komik:
    - Istilah terpanjang diproses terlebih dahulu.
    - Istilah yang diawali/diakhiri Katakana hanya diganti jika tidak menempel
      pada karakter Katakana lain (regex lookaround), mencegah 'ミナ' merusak 'ミナト'.
    - Istilah Kanji dan Hangul tetap memakai replace biasa karena partikel/honorifik
      menempel langsung (田中先生, 진우는).
    """
    if not text or not glossary:
        return text

    sorted_terms = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)

    for s_term, t_term in sorted_terms:
        if not s_term:
            continue

        is_kata_start = bool(re.match(rf'^{KATAKANA_CHAR}', s_term))
        is_kata_end = bool(re.search(rf'{KATAKANA_CHAR}$', s_term))

        if is_kata_start or is_kata_end:
            prefix = rf'(?<!{KATAKANA_CHAR})' if is_kata_start else ''
            suffix = rf'(?!{KATAKANA_CHAR})' if is_kata_end else ''
            pattern = rf'{prefix}{re.escape(s_term)}{suffix}'
            text = re.sub(pattern, t_term, text)
        else:
            text = text.replace(s_term, t_term)

    return text


def is_hallucinated_short(src: str, tgt: str) -> bool:
    """
    Guard cerdas halusinasi terjemahan pada teks pendek:
    - core = src tanpa spasi, angka, dan tanda baca; n = len(core).
    - Jika n == 0 atau n > 3: False (bukan teks pendek yang rentan halusinasi).
    - Jika core mengandung kanji: True bila kata >= 2n + 4.
    - Jika core katakana murni: True bila kata >= 3 (khas halusinasi SFX komik).
    - Selain itu: True bila kata >= 5.
    """
    if not src or not tgt:
        return False

    core = PUNCT_AND_DIGITS_REGEX.sub('', src)
    n = len(core)
    if n == 0 or n > 3:
        return False

    words = len(tgt.strip().split())

    # Jika mengandung kanji (kata Jepang bermakna padat seperti 大丈夫, 先生)
    if re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF]', core):
        return words >= (2 * n + 4)

    # Jika katakana murni (khas onomatopoeia/SFX seperti グリ, ゴゴゴ, ドン)
    if re.fullmatch(r'[\u30A0-\u30FF\uFF66-\uFF9Fー]+', core):
        return words >= 3

    # Pola lainnya (Hiragana murni pendek, dsb.)
    return words >= 5


def is_strong_name_candidate(cand: Dict[str, Any]) -> bool:
    """
    Memfilter kandidat ekstraksi nama diri:
    True HANYA jika auto_accept bernilai True, panjang istilah >= 2,
    memiliki suggested_translation, dan reasons beririsan dengan STRONG_NAME_REASONS.
    Katakana yang hanya berulang lintas halaman tanpa konteks nama tidak boleh lolos.
    """
    if not isinstance(cand, dict):
        return False

    if not cand.get("auto_accept"):
        return False

    term = cand.get("term", "")
    if len(term) < 2:
        return False

    sugg = cand.get("suggested_translation")
    if not sugg or not str(sugg).strip():
        return False

    raw_reasons = cand.get("reasons", set())
    if isinstance(raw_reasons, (list, tuple)):
        reasons_set = set(raw_reasons)
    elif isinstance(raw_reasons, set):
        reasons_set = raw_reasons
    else:
        reasons_set = {str(raw_reasons)}

    return bool(reasons_set & STRONG_NAME_REASONS)


def assess_translation(src: str, tgt: str) -> List[str]:
    """
    Menilai kualitas terjemahan dan menghasilkan daftar quality flags:
    - 'empty_translation': jika terjemahan kosong.
    - 'source_script_leak': jika keluaran masih bocor karakter asli (Kana/Han/Hangul).
    - 'possible_truncation': n >= 10 dan jumlah kata terjemahan < n / 6.
    - 'possible_overgeneration': n <= 6 dan kata terjemahan >= max(6, 2n).
    Hanya menghasilkan flag pelaporan, tidak pernah membuang hasil.
    """
    flags: List[str] = []

    if not tgt or not tgt.strip():
        flags.append("empty_translation")
        return flags

    # Kebocoran skrip asli sumber pada output
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uAC00-\uD7AF]', tgt):
        flags.append("source_script_leak")

    core = PUNCT_AND_DIGITS_REGEX.sub('', src)
    n = len(core)
    words = len(tgt.strip().split())

    if n >= 10 and words < (n / 6.0):
        flags.append("possible_truncation")

    if n <= 6 and n > 0 and words >= max(6, 2 * n):
        flags.append("possible_overgeneration")

    return flags
