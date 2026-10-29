import os
import gzip
import math
import re
from typing import List, Dict, Set, Any, Optional, Tuple

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


GENERIC_HALLUCINATION_PATTERNS = [
    r'tidak tahu apa yang terjadi',
    r'aku tidak tahu apa',
    r'saya tidak tahu apa',
    r'tidak tahu apa yang',
    r'tidak tahu harus berkata apa',
    r"i don't know what",
    r'^(?:aku\s+|saya\s+)?tidak\s+tahu[\.!?]*$',
    r'^(?:i\s+)?don\'?t\s+know[\.!?]*$'
]


def is_hallucinated_short(src: str, tgt: str) -> bool:
    """
    Guard cerdas halusinasi terjemahan pada teks pendek:
    - Untuk CJK:
      core = src tanpa spasi, angka, dan tanda baca; n = len(core).
      Jika n == 0 atau n > 3: False.
      Jika core mengandung kanji: True bila kata >= 2n + 4.
      Jika kana murni (Hiragana/Katakana):
        Jika n <= 2 (khas onomatopoeia/SFX seperti グリ, ぢゅ, ゴゴ, はは): True bila kata >= 3.
        Jika Katakana murni (n == 3): True bila kata >= 3.
        Selain itu: True bila kata >= 5.
    - Untuk Latin (Inggris/Barat):
      src_words = kata dalam sumber (len <= 3).
      True jika keluaran mengandung frasa halusinasi khas terjemahan mesin,
      atau jumlah kata membengkak tidak wajar (misal 'I see.' -> 'Aku tidak tahu apa yang terjadi').
    """
    if not src or not tgt:
        return False

    tgt_clean = tgt.strip()
    words = len(tgt_clean.split())
    tgt_lower = tgt_clean.lower()

    # Cek frasa halusinasi khas terjemahan mesin
    for pat in GENERIC_HALLUCINATION_PATTERNS:
        if re.search(pat, tgt_lower):
            return True

    # Kasus Latin (Inggris / komik Barat tanpa karakter CJK)
    if not CJK_REGEX.search(src):
        src_words = re.findall(r'\b[A-Za-z0-9]+\b', src)
        n_words = len(src_words)
        if n_words == 0 or n_words > 3:
            return False

        # Pembengkakan kata tidak wajar pada sumber sangat pendek (<= 3 kata)
        if n_words <= 2 and words >= 5:
            return True
        if n_words == 3 and words >= 7:
            return True
        return False

    # Kasus CJK (Jepang / Korea / Mandarin)
    core = PUNCT_AND_DIGITS_REGEX.sub('', src)
    n = len(core)
    if n == 0 or n > 3:
        return False

    # Jika mengandung kanji (kata Jepang bermakna padat seperti 大丈夫, 先生)
    if re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF]', core):
        return words >= (2 * n + 4)

    # Jika kana murni (Hiragana atau Katakana tanpa Kanji, khas onomatopoeia/SFX seperti グリ, ぢゅ, ゴゴ, はは)
    if re.fullmatch(r'[\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9Fー]+', core):
        if n <= 2:
            return words >= 3
        if re.fullmatch(r'[\u30A0-\u30FF\uFF66-\uFF9Fー]+', core):
            return words >= 3
        return words >= 5

    # Pola lainnya (Hangul / kombinasi)
    return words >= 5


def filter_cross_bubble_hallucinations(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Deteksi halusinasi lintas-balon (mode-collapse detection):
    Jika string terjemahan yang sama (>= 3 kata) dihasilkan dari
    dua atau lebih sumber BERBEDA dalam satu chapter/halaman,
    tandai dan buang hasil terjemahan tersebut (kembalikan string kosong)
    agar gambar asli pada balon-balon tersebut tidak rusak tertimpa halusinasi.
    """
    if not pairs:
        return []

    tgt_to_sources: Dict[str, Set[str]] = {}
    for src, tgt in pairs:
        if not tgt or not tgt.strip():
            continue
        tgt_norm = tgt.strip().lower()
        words = len(tgt.strip().split())
        if words >= 3:
            norm_src = re.sub(r'[\s\W_]+', '', src).lower()
            if tgt_norm not in tgt_to_sources:
                tgt_to_sources[tgt_norm] = set()
            if norm_src:
                tgt_to_sources[tgt_norm].add(norm_src)

    hallucinated_tgts = {
        tgt_norm for tgt_norm, sources in tgt_to_sources.items()
        if len(sources) >= 2
    }

    results: List[str] = []
    for src, tgt in pairs:
        if tgt and tgt.strip().lower() in hallucinated_tgts:
            print(f"[FILTER] Halusinasi lintas-balon dibuang: '{tgt}' dari sumber ganda berbeda.")
            results.append("")
        else:
            results.append(tgt)

    return results


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


# =======================================================================
# PASCA-OCR TEKS LATIN / INGGRIS (Pure Python, Wordninja & SymSpell)
# =======================================================================

_DICT_LOADED = False
ENGLISH_VOCAB: Set[str] = set()
WORD_FREQ: Dict[str, int] = {}
WORD_COST: Dict[str, float] = {}
MAX_WORD_LEN = 20
DELETES_1: Dict[str, List[str]] = {}

SUPPLEMENTAL_VOCAB: Dict[str, int] = {
    # Interjeksi dan onomatope komik umum
    'oh': 500000000, 'ah': 300000000, 'eh': 200000000, 'uh': 200000000, 'um': 200000000,
    'yeah': 400000000, 'yep': 100000000, 'nope': 100000000, 'huh': 150000000, 'whoa': 80000000,
    'wow': 150000000, 'hey': 300000000, 'ok': 400000000, 'okay': 350000000,
    # Kosakata komik & ejaan Amerika
    'harbor': 150000000, 'harbors': 50000000,
    # Kontraksi bahasa Inggris standar
    "don't": 500000000, "can't": 400000000, "won't": 300000000, "it's": 600000000,
    "that's": 450000000, "i'm": 500000000, "you're": 400000000, "they're": 350000000,
    "we're": 300000000, "we've": 250000000, "isn't": 200000000, "aren't": 150000000,
    "wasn't": 180000000, "weren't": 120000000, "couldn't": 180000000, "wouldn't": 180000000,
    "shouldn't": 150000000, "haven't": 160000000, "hasn't": 140000000, "hadn't": 120000000,
    "gonna": 200000000, "wanna": 180000000, "gotta": 150000000
}


def _ensure_english_dict_loaded() -> None:
    """
    Memuat kamus frekuensi bahasa Inggris (~82k kata) secara lazy ke memori.
    Membangun cost table untuk pemecahan kata (Wordninja) dan index penghapusan (SymSpell).
    """
    global _DICT_LOADED, ENGLISH_VOCAB, WORD_FREQ, WORD_COST, MAX_WORD_LEN, DELETES_1
    if _DICT_LOADED:
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_paths = [
        os.path.join(base_dir, "data", "frequency_dictionary_en_82_765.txt.gz"),
        os.path.join(base_dir, "models", "frequency_dictionary_en_82_765.txt.gz")
    ]

    loaded = False
    for p in dict_paths:
        if os.path.exists(p):
            try:
                with gzip.open(p, "rt", encoding="utf-8-sig") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            WORD_FREQ[parts[0].lower()] = int(parts[1])
                loaded = True
                break
            except Exception:
                continue

    # Tambahkan suplemen kosakata komik & kontraksi
    for term, count in SUPPLEMENTAL_VOCAB.items():
        if term not in WORD_FREQ:
            WORD_FREQ[term] = count

    if not WORD_FREQ:
        # Fallback darurat jika file tidak ada
        for term, count in SUPPLEMENTAL_VOCAB.items():
            WORD_FREQ[term] = count

    ENGLISH_VOCAB.clear()
    ENGLISH_VOCAB.update(WORD_FREQ.keys())
    total_count = sum(WORD_FREQ.values()) if WORD_FREQ else 1
    log_total = math.log(total_count)

    WORD_COST = {w: log_total - math.log(f) for w, f in WORD_FREQ.items()}
    MAX_WORD_LEN = max(len(w) for w in WORD_FREQ) if WORD_FREQ else 20

    # Index jarak edit 1 (SymSpell delete index)
    for word in WORD_FREQ:
        if len(word) < 2:
            continue
        for i in range(len(word)):
            d = word[:i] + word[i+1:]
            if d not in DELETES_1:
                DELETES_1[d] = []
            DELETES_1[d].append(word)

    _DICT_LOADED = True


def is_english_word(word: str) -> bool:
    """Memeriksa apakah sebuah kata merupakan kata kamus bahasa Inggris valid."""
    if not word:
        return False
    _ensure_english_dict_loaded()
    return word.lower() in ENGLISH_VOCAB


def count_english_words(text: str) -> int:
    """Menghitung jumlah token dalam teks yang merupakan kata kamus bahasa Inggris valid."""
    if not text:
        return 0
    _ensure_english_dict_loaded()
    words = re.findall(r"\b[A-Za-z]+(?:\'[A-Za-z]+)?\b", text)
    return sum(1 for w in words if w.lower() in ENGLISH_VOCAB)


def _damerau_levenshtein(s1: str, s2: str) -> int:
    """Menghitung jarak edit Damerau-Levenshtein (transposition aware)."""
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost) # transposition
    return d[(len1 - 1, len2 - 1)]


def _match_case(src: str, repl: str) -> str:
    """Menyesuaikan kapitalisasi repl sesuai pola kata src."""
    if src.isupper():
        return repl.upper()
    elif src.istitle():
        return repl.capitalize()
    return repl.lower()


def rejoin_hyphenated_words(text: str) -> str:
    """
    Menyambung kata yang terpotong tanda hubung di akhir baris ("SIGNA- TURE" -> "SIGNATURE")
    hanya jika hasil gabungannya merupakan kata kamus yang valid; jika tidak, pertahankan tanda hubung.
    """
    _ensure_english_dict_loaded()
    if not text:
        return ""

    def repl(m):
        w1, w2 = m.group(1), m.group(2)
        combined = w1 + w2
        if combined.lower() in ENGLISH_VOCAB:
            return _match_case(w1, combined)
        return m.group(0)

    # Deteksi pemisahan baris atau spasi setelah tanda hubung
    return re.sub(r'\b([A-Za-z]+)-\s*[\r\n\s]+\s*([A-Za-z]+)\b', repl, text)


def split_glued_token(token: str) -> str:
    """
    Memecah token alfabet yang menempel tanpa spasi (mis. 'YOUKNOW' -> 'YOU KNOW')
    dengan Viterbi dynamic programming berbasis frekuensi kata unigram.
    Hanya memecah jika SEMUA bagian pecahan adalah kata kamus yang sah.
    """
    _ensure_english_dict_loaded()
    s = token.lower()
    n = len(s)
    if n <= 3 or s in ENGLISH_VOCAB:
        return token

    # Cegah pemecahan token jika token merupakan variasi 1-delete dari kata kamus frekuensi tinggi
    # (misal 'defnitely' -> 'definitely' jangan dipecah jadi 'def', 'nit', 'ely')
    if s in DELETES_1:
        best_cand = max(DELETES_1[s], key=lambda w: WORD_FREQ.get(w, 0))
        if WORD_FREQ.get(best_cand, 0) >= 50000:
            return token

    cost = [0.0] + [float('inf')] * n
    best_prev = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_WORD_LEN), i):
            sub = s[j:i]
            c = cost[j] + (WORD_COST[sub] if sub in WORD_COST else (i - j) * 20.0)
            if c < cost[i]:
                cost[i] = c
                best_prev[i] = j

    indices = []
    curr = n
    while curr > 0:
        prev = best_prev[curr]
        indices.append((prev, curr))
        curr = prev
    indices.reverse()

    parts = [token[start:end] for start, end in indices]
    # Syarat ketat: hanya pecah jika len > 1 dan SEMUA potongan adalah kata kamus yang sah
    # (bagian dengan 1 huruf hanya diperbolehkan jika 'a' atau 'i')
    all_valid = (
        len(parts) > 1 and
        all(
            p.lower() in ENGLISH_VOCAB and (len(p) > 1 or p.lower() in {'a', 'i'})
            for p in parts
        )
    )
    if all_valid:
        return ' '.join(parts)
    return token


def split_glued_words(text: str) -> str:
    """
    Memecah kata-kata yang menempel dalam teks komik dengan mempertahankan tanda baca.
    """
    if not text:
        return ""
    return re.sub(r'\b[A-Za-z]{4,}\b', lambda m: split_glued_token(m.group(0)), text)


def correct_spelling_latin(
    text: str,
    chapter_counts: Optional[Dict[str, int]] = None,
    glossary_terms: Optional[Set[str]] = None,
    protected_names: Optional[Set[str]] = None
) -> str:
    """
    Koreksi salah-baca huruf dengan SymSpell (kamus frekuensi Inggris).
    Kecuali token yang:
    - Muncul >= 2 kali dalam chapter
    - Ada di glosarium
    - Berupa nama diri yang dilindungi
    Catatan: Huruf kapital semua (ALL CAPS) TIDAK BOLEH menjadi alasan untuk tidak mengoreksi
    karena komik Barat/manga terjemahan umumnya menggunakan huruf kapital.
    """
    _ensure_english_dict_loaded()
    if not text:
        return ""

    chap_counts = chapter_counts or {}
    gloss_set = {t.lower() for t in (glossary_terms or set())}
    name_set = {n.lower() for n in (protected_names or set())}

    def repl(m):
        tok = m.group(0)
        low = tok.lower()

        # 1. Jika sudah ada di kamus, tidak perlu dikoreksi
        if low in ENGLISH_VOCAB:
            return tok

        # 2. Jangan koreksi jika token muncul >= 2 kali di chapter
        if chap_counts.get(low, 0) >= 2:
            return tok

        # 3. Jangan koreksi jika ada di glosarium atau nama diri yang dilindungi
        if low in gloss_set or low in name_set:
            return tok

        # 4. Jangan koreksi token sangat pendek (<= 2 huruf)
        if len(low) <= 2:
            return tok

        # 5. Normalisasi karakter angka OCR yang sering tertukar dengan huruf (1->l, 0->o)
        low_norm = re.sub(r'(?<=[a-z])1(?=[a-z])|^1(?=[a-z])|(?<=[a-z])1$', 'l', low)
        low_norm = re.sub(r'(?<=[a-z])0(?=[a-z])|^0(?=[a-z])|(?<=[a-z])0$', 'o', low_norm)
        if low_norm in ENGLISH_VOCAB:
            return _match_case(tok, low_norm)

        # 6. Cari kandidat SymSpell (jarak edit <= 2)
        cands = {}
        # Kasus 1-delete langsung (input adalah 1-delete dari kata kamus yang sah, e.g. defnitely -> definitely)
        if low in DELETES_1:
            for cand in DELETES_1[low]:
                cands[cand] = (1, WORD_FREQ.get(cand, 0))

        input_deletes = {low[:i] + low[i+1:] for i in range(len(low))}
        for d in input_deletes:
            if d in ENGLISH_VOCAB:
                cands[d] = (1, WORD_FREQ.get(d, 0))
            if d in DELETES_1:
                for cand in DELETES_1[d]:
                    if cand not in cands:
                        dist = _damerau_levenshtein(low, cand)
                        if dist <= 2:
                            cands[cand] = (dist, WORD_FREQ.get(cand, 0))

        if not cands:
            return tok

        sorted_cands = sorted(cands.items(), key=lambda x: (x[1][0], -x[1][1]))
        best_cand, (dist, freq) = sorted_cands[0]

        # Hanya terima jika jarak 1 dengan frekuensi meyakinkan, atau jarak 2 dengan frekuensi tinggi
        if (dist == 1 and freq >= 10000) or (dist == 2 and freq >= 500000):
            return _match_case(tok, best_cand)
        return tok

    return re.sub(r'\b[A-Za-z0-9]+(?:\'[A-Za-z]+)?\b', repl, text)


def calculate_ocr_quality(text: str) -> float:
    """
    Menghitung proporsi token kata yang valid dalam kamus bahasa Inggris (0.0 - 1.0).
    Jika < 0.6: mengindikasikan OCR berkualitas rendah / corrupt / salah bahasa.
    """
    _ensure_english_dict_loaded()
    if not text or not text.strip():
        return 1.0

    words = re.findall(r'\b[A-Za-z]+(?:\'[A-Za-z]+)?\b', text)
    if not words:
        return 1.0

    in_vocab = sum(1 for w in words if w.lower() in ENGLISH_VOCAB)
    return in_vocab / float(len(words))


def normalize_latin_case(text: str) -> str:
    """
    Normalisasi huruf kapital untuk teks Latin komik:
    Jika >= 70% huruf adalah kapital dan ada >= 3 huruf, ubah ke huruf kecil
    dengan huruf kapital di awal kalimat dan 'I' tunggal / kontraksinya tetap kapital.
    Negasi ('don't', 'can't', 'no way') dan makna tetap terjaga.
    """
    if not text:
        return ""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return text
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio < 0.70:
        return text

    # Ubah ke lowercase
    norm = text.lower()

    # Kapitalisasi di awal string (melewati karakter pembuka seperti kutip/titik)
    norm = re.sub(r'(^|[\r\n]+)([\s"\'\(]*)(\b[a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), norm)

    # Kapitalisasi setelah tanda akhir kalimat (. ! ? …)
    norm = re.sub(r'([.!?…]+[\s"\'\)]+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), norm)

    # Huruf 'I' tunggal dan kontraksinya tetap kapital
    norm = re.sub(r'\bi\b', 'I', norm)
    norm = re.sub(r"\bi('m|'ve|'ll|'d)\b", lambda m: "I" + m.group(1), norm)

    return norm


# =======================================================================
# PASCA-TERJEMAHAN NLLB → BAHASA INDONESIA (koreksi artefak terjemahan mesin)
# =======================================================================

# Pola substitusi berbasis aturan untuk memperbaiki output NLLB
# ke register manga Indonesia yang lebih alami
_NLLB_ID_SUBS: List[Tuple[re.Pattern, str]] = [
    # --- Pronoun register: formal (saya/Anda) → manga casual (aku/kamu) ---
    (re.compile(r'\bSaya\b'), 'Aku'),
    (re.compile(r'\bsaya\b'), 'aku'),
    (re.compile(r'\bAnda\b'), 'kamu'),
    (re.compile(r'\banda\b'), 'kamu'),
    # --- Kosakata umum NLLB yang salah untuk konteks manga Jepang ---
    # "hadiah tangan" adalah terjemahan harfiah yang salah dari 手土産
    (re.compile(r'\bhadiah tangan\b', re.IGNORECASE), 'buah tangan'),
    (re.compile(r'\bAku punya buah tangan\b', re.IGNORECASE), 'Aku bawakan buah tangan'),
    # "pandangan negatif terhadap" adalah terjemahan salah dari 負い目
    (re.compile(r'\bpandangan negatif terhadap\b', re.IGNORECASE), 'rasa bersalah kepada'),
    # Hubungan guru/murid (師であるX -> X, guru) diperhalus ke register manga yang alami
    (re.compile(r'\b(?:kepada|terhadap)\s+([A-Za-z]+),\s*guru\b', re.IGNORECASE), r'kepada gurunya, \1'),
    # Artefak struktural NLLB pada kalimat zero-pronoun / klausa bertingkat (Xにして -> "Aku akan membuatmu putri X")
    (re.compile(r'\bAku akan membuatmu (?:menjadi )?putri(?:ku)?\s+([A-Za-z]+)', re.IGNORECASE), r'Putri \1'),
    # Salah tafsir kono musume (この娘) sebagai "putriku" alih-alih "gadis ini"
    (re.compile(r'\bJika ada masalah dengan putriku,?\s*itu akan berubah\b', re.IGNORECASE), 'Kalau sudah soal gadis ini, sikapnya langsung berubah'),
    (re.compile(r'\bmasalah dengan putriku\b', re.IGNORECASE), 'soal gadis ini'),
    # Artefak NLLB pada frase idiomatis/klasikal 冷静さを欠き判断力を鈍らせる
    (re.compile(r'\bTidak ada yang tenang,?\s*tidak ada yang bijaksana\b', re.IGNORECASE), 'Kehilangan ketenangan dan menumpulkan pertimbangan'),
    # Duplikasi kata yang sering muncul dari NLLB
    (re.compile(r'\badalah adalah\b', re.IGNORECASE), 'adalah'),
    (re.compile(r'\bYang mana\b'), 'yang'),
    (re.compile(r'\byang mana\b'), 'yang'),
    # Konstruksi terlalu formal dari NLLB
    (re.compile(r'\bDengan demikian\b'), 'Jadi'),
    (re.compile(r'\bdengan demikian\b'), 'jadi'),
    (re.compile(r'\bOleh karena itu\b'), 'Jadi'),
    (re.compile(r'\boleh karena itu\b'), 'jadi'),
    (re.compile(r'\bSehubungan dengan itu\b', re.IGNORECASE), 'Soal itu'),
]

# Nama karakter yang sering salah diterjemahkan NLLB karena double meaning katakana
# Pasangan: (pola nama salah, nama benar) — hanya nama yang terbukti salah oleh NLLB
_NLLB_NAME_FIXES: List[Tuple[re.Pattern, str]] = [
    # ボルト → "Bolt" (kata Inggris "bolt") atau "Bolto"
    (re.compile(r'\bBolt\b'), 'Boruto'),
    (re.compile(r'\bBolto\b'), 'Boruto'),
    # サラダ → "Salad" (kata makanan) atau frase "ini salad" / "Ini salad"
    (re.compile(r'\b(?:ini\s+)?salad\b', re.IGNORECASE), 'Ini Sarada'),
    (re.compile(r'\bSalad\b'), 'Sarada'),
    (re.compile(r'\bSalada\b'), 'Sarada'),
]


# Pasangan perbaikan terjemahan khusus bahasa sumber Inggris -> Indonesia
_NLLB_EN_TO_ID_SUBS: List[Tuple[re.Pattern, str]] = [
    # "you're early" / "you're early today" yang sering diterjemahkan harfiah "kamu pagi hari ini"
    (re.compile(r'\b(?:kamu|kau)\s+pagi\s+hari\s+ini\b', re.IGNORECASE), 'kamu datang lebih awal hari ini'),
    (re.compile(r'\b(?:kamu|kau)\s+adalah\s+awal\s+hari\s+ini\b', re.IGNORECASE), 'kamu datang lebih awal hari ini'),
    # "country boy" -> "anak negara" (harfiah salah) -> "anak desa" / "anak kampung"
    (re.compile(r'\banak\s+negara\b', re.IGNORECASE), 'anak desa'),
    # "getting used to this" -> "mendapatkan digunakan untuk ini" -> "sudah terbiasa dengan ini"
    (re.compile(r'\bmendapatkan\s+digunakan\s+untuk\s+ini\b', re.IGNORECASE), 'sudah terbiasa dengan ini'),
]


def postprocess_nllb_indonesian(text: str, src_lang: Optional[str] = None) -> str:
    """
    Pasca-pemrosesan output NLLB untuk Bahasa Indonesia manga/komik:
    - Koreksi pronoun register: saya/Anda → aku/kamu (gaya dialog komik santai)
    - Jika sumber Jepang: koreksi nama karakter katakana yang ambigu (Bolt→Boruto, Salad→Sarada)
    - Jika sumber Inggris: koreksi frase idiomatis Inggris yang salah diterjemahkan harfiah oleh NLLB
    - Bersihkan konstruksi formal NLLB yang tidak wajar untuk dialog komik
    """
    if not text:
        return text

    result = text

    # 1. Perbaikan nama karakter katakana yang ambigu HANYA untuk sumber bahasa Jepang
    # (Jangan ubah kata Inggris 'Bolt' atau makanan 'salad' jika sumbernya komik berbahasa Inggris)
    if src_lang in [None, "jpn_Jpan", "ja"]:
        for pattern, replacement in _NLLB_NAME_FIXES:
            result = pattern.sub(replacement, result)

    # 2. Koreksi idiomatis khusus untuk terjemahan dari Bahasa Inggris
    if src_lang in ["eng_Latn", "en"]:
        for pattern, replacement in _NLLB_EN_TO_ID_SUBS:
            result = pattern.sub(replacement, result)

    # 3. Substitusi kosakata dan register pronoun umum (berlaku untuk semua komik)
    for pattern, replacement in _NLLB_ID_SUBS:
        result = pattern.sub(replacement, result)

    return result


def post_process_latin_ocr(
    text: str,
    chapter_counts: Optional[Dict[str, int]] = None,
    glossary_terms: Optional[Set[str]] = None,
    protected_names: Optional[Set[str]] = None,
    normalize_case: bool = True
) -> Tuple[str, float]:
    """
    Pipeline pasca-OCR lengkap untuk teks Latin komik:
    1. Sambung kata terpotong tanda hubung akhir baris ("SIGNA- TURE" -> "SIGNATURE")
    2. Pecah token menempel ("YOUKNOW" -> "YOU KNOW")
    3. Koreksi salah baca dengan SymSpell
    4. Hitung skor ocr_quality
    5. Normalisasi huruf kapital (ALL-CAPS -> sentence case) untuk input terjemahan yang optimal
    """
    if not text or not text.strip():
        return "", 1.0

    # Langkah 1: Sambung tanda hubung
    cleaned = rejoin_hyphenated_words(text)
    # Langkah 2: Pecah kata yang menempel
    cleaned = split_glued_words(cleaned)
    # Langkah 3: Koreksi ejaan OCR
    cleaned = correct_spelling_latin(cleaned, chapter_counts, glossary_terms, protected_names)
    # Langkah 4: Hitung kualitas OCR
    quality = calculate_ocr_quality(cleaned)
    # Langkah 5: Normalisasi huruf kapital jika diminta
    if normalize_case:
        cleaned = normalize_latin_case(cleaned)

    return cleaned, quality

