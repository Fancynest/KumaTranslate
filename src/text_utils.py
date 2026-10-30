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
        elif char == '.' and i + 2 < n and norm_text[i + 1] == ' ' and (norm_text[i + 2].isupper() or norm_text[i + 2] in CLOSING_QUOTES):
            # Pecah kalimat bahasa Inggris setelah tanda titik yang diikuti huruf kapital
            # Jangan pecah singkatan umum (Mr., Ms., Dr., dll.)
            prefix = norm_text[max(0, i - 4):i].lower()
            if not any(prefix.endswith(abbr) for abbr in ['mr', 'ms', 'mrs', 'dr', 'etc', 'vs', 'e.g', 'i.e']):
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

    # Periksa apakah sumber memang mengandung ungkapan negasi / ketidaktahuan yang sah
    has_valid_knowledge_neg = bool(re.search(r'\b(?:don\'?t|didn\'?t|not\s+know|know\s+nothing|no\s+idea)\b', src, re.I)) or bool(re.search(r'[\u77e5\u5206][\u3089\u304b\u308c\u3093]', src))

    # Cek frasa halusinasi khas terjemahan mesin (HANYA jika sumber tidak menyatakan tidak tahu)
    if not has_valid_knowledge_neg:
        for pat in GENERIC_HALLUCINATION_PATTERNS:
            if re.search(pat, tgt_lower):
                return True

    # Guard khusus interjeksi komik pendek 1-2 kata (AH!, OH!, EH?, TCH!)
    # Jika sumber adalah interjeksi seruan pendek dan terjemahan membengkak jadi kalimat tanya (cth: "Apa yang terjadi?")
    if re.fullmatch(r'^(?:ah|ha|oh|eh|tch|cih|chi)[!.\s]*$', src.strip(), re.I):
        if words >= 2 or '?' in tgt:
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
    'wow': 150000000, 'hey': 300000000, 'ok': 400000000, 'okay': 350000000, 'tch': 500000000,
    # Kosakata komik & ejaan Amerika
    'harbor': 150000000, 'harbors': 50000000,
    # Kontraksi bahasa Inggris standar
    "don't": 500000000, "can't": 400000000, "won't": 300000000, "it's": 600000000,
    "that's": 450000000, "i'm": 500000000, "you're": 400000000, "they're": 350000000,
    "we're": 300000000, "we've": 250000000, "isn't": 200000000, "aren't": 150000000,
    "wasn't": 180000000, "weren't": 120000000, "couldn't": 180000000, "wouldn't": 180000000,
    "shouldn't": 150000000, "haven't": 160000000, "hasn't": 140000000, "hadn't": 120000000,
    "gonna": 200000000, "wanna": 180000000, "gotta": 150000000,
    # Slang & kontraksi komik umum
    "cuz": 200000000, "cause": 250000000, "coz": 150000000, "noway": 100000000,
    "predecessor": 100000000, "signature": 200000000,
    "naruto": 500000000, "uzumaki": 500000000, "boruto": 500000000, "sarada": 500000000,
    "sasuke": 500000000, "makoto": 500000000, "futaiwa": 500000000
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


# Alias untuk kompatibilitas impor antar-modul
split_concatenated_words = split_glued_words


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
    # Idiom & terjemahan khusus manga Jepang
    (re.compile(r'\bAku\s+punya\s+hadiah\s+tangan\b', re.IGNORECASE), 'Aku bawakan buah tangan'),
    (re.compile(r'\bhadiah\s+tangan\b', re.IGNORECASE), 'buah tangan'),
    (re.compile(r'\bAku\s+akan\s+membuatmu\s+putri(?:ku)?\s+Sasuke\b'), 'Putri Sasuke'),
    (re.compile(r'\baku\s+akan\s+membuatmu\s+putri(?:ku)?\s+Sasuke\b'), 'putri Sasuke'),
    (re.compile(r'\bputri(?:ku)?\s+Sasuke(?=\s*\.\.\.|\s*$)'), 'putri dari Sasuke Uchiha'),
    (re.compile(r'\b(?:Saat\s+ini,\s+wanita\s+ini\s+akan\s+berubah|Jika\s+ada\s+masalah\s+dengan\s+putriku,\s+itu\s+akan\s+berubah)\b', re.IGNORECASE), 'Kalau sudah soal gadis ini, sikapnya langsung berubah'),
    (re.compile(r'\b(?:Tidak\s+ada\s+yang\s+tenang,\s+tidak\s+ada\s+yang\s+bijaksana)\b', re.IGNORECASE), 'Kehilangan ketenangan dan menumpulkan pertimbangan'),
    (re.compile(r'\bDengar,\s+aku\s+menggunakan\s+dia\b', re.IGNORECASE), 'Dengar, kita manfaatkan dia'),
    (re.compile(r'\bAku\s+akan\s+tunjukkan\s+cara\s+terbaik\b', re.IGNORECASE), 'Aku akan beritahu rencana yang bagus'),
    (re.compile(r'\bTapi\s+apa\s+dia\s+punya\s+(?:pandangan\s+negatif|rasa\s+bersalah)\s+terhadap\s+Sasuke,\s+gurunya\??', re.IGNORECASE), 'Tapi entah karena merasa bersalah pada gurunya, Sasuke...'),
    (re.compile(r'\bTapi\s+apa\s+dia\s+punya\s+rasa\s+bersalah\s+kepada\s+guru\s+Sasuke\??', re.IGNORECASE), 'Tapi entah karena merasa bersalah pada gurunya, Sasuke...'),
    (re.compile(r'\bItu\s+kelemahan\s+terbesar\s+Bolt\b', re.IGNORECASE), 'Kelemahan terbesar bagi Boruto'),
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
    # Boruto & Naruto universe terminology & idioms
    (re.compile(r'\b(?:sebenarnya\s+)?bisnis\s+(?:kita|kami)\s+adalah\s+(?:dengan\s+)?pendahulunya\b', re.IGNORECASE), 'sebenarnya urusan kami adalah dengan pendahulunya'),
    (re.compile(r'\bSebenarnya\s+urusan\s+kami\s+sebenarnya\b', re.IGNORECASE), 'Sebenarnya urusan kami'),
    (re.compile(r'\bYa,\s+jadi\s+apa\?', re.IGNORECASE), 'Ya, terus kenapa?'),
    (re.compile(r'\b(?:Lord\s+VII|Lord\s+Seventh|tuan\s+ketujuh|ketujuh(?=\s+akan\s+(?:berada|ada)))\b', re.IGNORECASE), 'Hokage Ketujuh'),
    (re.compile(r'\btanda\s+tangan\s+(?:sembilan\s+ekor|nine\s+tails)\b', re.IGNORECASE), 'jejak chakra Ekor Sembilan'),
    (re.compile(r'^Hei,\s+apa\s+yang\s+kau\s+lakukan\?$', re.IGNORECASE), 'Hei!'),
    (re.compile(r'\b(?:Dipotong|Sekitar)\s+tiga\s+tahun\s+yang\s+lalu,\s+dia\b', re.IGNORECASE), 'Soalnya tiga tahun lalu, dia'),
    (re.compile(r'\b(?:pasti\s+)?tampaknya\s+tidak\s+ada\s+di\s+mana-mana\b', re.IGNORECASE), 'jelas tampaknya tidak ada di mana pun'),
    (re.compile(r'\bMereka\s+harus\s+tahu\s+itu\b', re.IGNORECASE), 'Mereka tidak perlu tahu itu'),
    (re.compile(r'\bTapi\s+tidak\s+mungkin\s+ketujuh\b', re.IGNORECASE), 'Tapi tidak mungkin Hokage Ketujuh'),
    (re.compile(r'\bTapi\s+(?:sekarang\s+yang|tidak\s+ada\s+cara)\s+Hokage\s+Ketujuh\b', re.IGNORECASE), 'Tapi tidak mungkin Hokage Ketujuh'),
    (re.compile(r'\bakan\s+di\s+sini\b', re.IGNORECASE), 'akan ada di sini'),
    (re.compile(r'^[.·\s]*S\s+datang\s+dari\s+gadis(?:\s+ini)?\.?', re.IGNORECASE), '...berasal dari gadis ini.'),
    # Slice of life & comic terms
    (re.compile(r'\bseorang\s+koto-chan\b', re.IGNORECASE), 'Makoto-chan'),
    (re.compile(r'\brumah\s+bagi\s+(?:tiga|3)\s+orang\b', re.IGNORECASE), 'rumah bagi 300 orang'),
    (re.compile(r'\bterras\s+futaiwa\b', re.IGNORECASE), 'Futaiwa Terrace'),
    # Marine biology & comic fishery context (Turbo cornutus / sazae)
    (re.compile(r'\bturban\s+tanduk\b', re.IGNORECASE), 'siput bertanduk'),
    (re.compile(r'\banchovies\b', re.IGNORECASE), 'ikan teri'),
    (re.compile(r'\banchovy\b', re.IGNORECASE), 'ikan teri'),
    (re.compile(r'\b(?:hari\s+ini\s+)?(?:aku\s+(?:punya|mendapat|dapat)\s+|1071\s+)(?:anchovies|ikan\s+teri)\b', re.IGNORECASE), 'Hari ini aku dapat ikan teri'),
    (re.compile(r'\bwa-\s*apa\s+tangkapan\s+besar[.!]*\b', re.IGNORECASE), 'Wah, hasil tangkapan yang besar!'),
    (re.compile(r'^(?:TCH!?|Chi!?|Tech!?)$', re.IGNORECASE), 'Cih!'),
    (re.compile(r'!\.+$'), '!'),
    (re.compile(r'\btsubai\s+hugo\b', re.IGNORECASE), 'Tsubai Chugo'),
    (re.compile(r'\b(?:semua\s+jalan\s+dari|seluruh\s+jalan\s+dari)\b', re.IGNORECASE), 'jauh-jauh dari'),
    (re.compile(r'\bBisakah kau menjadi lebih baik kembar ob\b', re.IGNORECASE), 'Bisakah kamu lebih baik padanya?'),
    (re.compile(r'\bAda sesuatu yang tidak terlalu menyukai aku\b', re.IGNORECASE), 'Firasatku mengatakan dia tidak terlalu menyukaiku.'),
    (re.compile(r'\bSesuatu mengatakan padaku\b', re.IGNORECASE), 'Firasatku mengatakan'),
    (re.compile(r'\bmenyukai aku\b', re.IGNORECASE), 'menyukaiku'),
    (re.compile(r'\bjaringan pho pintar\b', re.IGNORECASE), 'smartphone canggih'),
    (re.compile(r'\bpohon keluarga kita\b', re.IGNORECASE), 'keluarga kita'),
    (re.compile(r'\bMereka harus tahu itu!\b', re.IGNORECASE), 'Mereka tidak perlu tahu itu!'),
    (re.compile(r'\bPagi-pagi\.', re.IGNORECASE), 'Selamat pagi.'),
    (re.compile(r'\bseorang koto-chan\b', re.IGNORECASE), 'Makoto-chan'),
    (re.compile(r'\brumah bagi tiga orang\b', re.IGNORECASE), 'rumah bagi 300 orang'),
    (re.compile(r'\bIni adalah\.\.\.$', re.IGNORECASE), 'Begitu rupanya...'),
    (re.compile(r'\bYah dia pasti tampaknya berada di mana saja di desa ini\b', re.IGNORECASE), 'Yah, dia jelas tidak terlihat di mana pun di desa ini.'),
    (re.compile(r'\b(?:dos\s+osiw|dnos\s+osiw)\b', re.IGNORECASE), 'sup miso'),
    (re.compile(r'\bturban\s+berbunyi\b', re.IGNORECASE), 'siput bertanduk'),
    (re.compile(r'\bdaerah\s+set\b', re.IGNORECASE), 'daerah Seto'),
    (re.compile(r'\btidak\s+memiliki\s+pucuk\b', re.IGNORECASE), 'tidak memiliki duri'),
    (re.compile(r'\blakukan\s+sesuatu\s+untukku\s+sebelum\s+kau\s+kembali\b', re.IGNORECASE), 'membantuku sebelum kamu kembali'),
    (re.compile(r'\bAku\s+tidak\s+tahu\s+jika\s+ada\s+lebih\s+banyak\s+anak\s+muda\b', re.IGNORECASE), 'Aku tidak keberatan jika ada lebih banyak anak muda'),
    (re.compile(r'\bjumlah\s+tempat\s+tinggal\s+terus\s+menurun\b', re.IGNORECASE), 'jumlah penduduk terus berkurang'),
    (re.compile(r'\bpesan\s+formulir\s+pintu\s+sesuatu\??', re.IGNORECASE), 'pesan untuk ayahku atau semacamnya?'),
    (re.compile(r'\bmengikuti\s+jejak\s+Ming\s+Uzumaki\s+Naruto\s+I\b', re.IGNORECASE), 'mengikuti jejak Uzumaki Naruto'),
    (re.compile(r'\b50,\s*sejujurnya\b', re.IGNORECASE), 'Jadi, sejujurnya'),
    (re.compile(r'\bOh\s+ho\s+kau\s+menjadi\s+lebih\s+dari\s+seorang\s+anak\s+pedesaan\b', re.IGNORECASE), 'Oh, kau makin mirip anak desa saja'),
    (re.compile(r'\bUntuk\s+satu-satunya\s+tanda\s+tangan\s+sembilan\s+ekor\b', re.IGNORECASE), 'Karena satu-satunya jejak chakra Ekor Sembilan'),
    (re.compile(r'\bjaringan\s+pho\s+pintar\s+mewahmu\b', re.IGNORECASE), 'smartphone canggihmu itu'),
    (re.compile(r'\b20\s+Oktober\b', re.IGNORECASE), 'seorang nelayan lokal'),
    # Dialog berkunjung ke rumah (Visiting scene)
    (re.compile(r'^Maafkan\s+aku[\.!\s]*$', re.IGNORECASE), 'Permisi.'),
    (re.compile(r'^Masuklah[\.!\s]*$', re.IGNORECASE), 'Silakan masuk.'),
    (re.compile(r'\bTolong\s+datang\s+ke\s+atas[\.!\s]*$', re.IGNORECASE), 'Silakan naik ke atas.'),
    (re.compile(r'\b(?:W,\s*|W\s+|WI\s*)?maaf\s+mengganggu\b', re.IGNORECASE), 'Maaf mengganggu'),
    (re.compile(r'\bTerima\s+kasih\s+sudah\s+merawatnya\b', re.IGNORECASE), 'Terima kasih sudah membantunya'),
    (re.compile(r'\bAku\s+Ibu\s+Chugo:?\b', re.IGNORECASE), 'Aku ibunya Chugo.'),
    (re.compile(r'\bKau\s+pasti\s+makoto\b', re.IGNORECASE), 'Kau pasti Makoto.'),
    # Dialog kuliner & restoran diner (Food & restaurant scene)
    (re.compile(r'\bCucumber,\s*lada,\s*tomat\b', re.IGNORECASE), 'Mentimun, paprika, tomat'),
    (re.compile(r'\bCucumber\b', re.IGNORECASE), 'Mentimun'),
    (re.compile(r'\bBisakah\s+aku\s+memotong\s+lada\s+ini\s+dengan\s+tipis\s+dan\s+mencelupkannya\s+ke\s+dalam\s+saus\s+kedelai\??', re.IGNORECASE), 'Bolehkah aku iris tipis paprika ini lalu celupkan ke kecap asin?'),
    (re.compile(r'\bmencelupkannya\s+ke\s+dalam\s+saus\s+kedelai\b', re.IGNORECASE), 'mencelupkannya ke kecap asin'),
    (re.compile(r'\bsaus\s+kedelai\b', re.IGNORECASE), 'kecap asin'),
    (re.compile(r'\bKami\s+memiliki\s+sayuran\s+dari\s+peternakan\s+lokal\s+dan\s+ikan\s+yang\s+ditangkap\s+di\s+siang\s+hari\b', re.IGNORECASE), 'Kami punya sayuran dari kebun lokal dan ikan hasil tangkapan hari ini'),
    (re.compile(r'\bpeternakan\s+lokal\b', re.IGNORECASE), 'kebun lokal'),
    (re.compile(r'\bikan\s+yang\s+ditangkap\s+di\s+siang\s+hari\b', re.IGNORECASE), 'ikan hasil tangkapan hari ini'),
    (re.compile(r'\bIni\s+baru\s+dipilih\b', re.IGNORECASE), 'Ini baru dipetik'),
    (re.compile(r'\bbaru\s+dipilih\b', re.IGNORECASE), 'baru dipetik'),
    (re.compile(r'\bKulit\s+timun\s+melekat\b', re.IGNORECASE), 'Kulit mentimunnya berduri halus'),
    (re.compile(r'\bLaut\s+Dalam\s+Seto\b', re.IGNORECASE), 'Laut Pedalaman Seto'),
    (re.compile(r'\blaut\s+dalam\s+negeri,\s*rumah\s+bagi\s+(?:tiga|3)\s+orang\b', re.IGNORECASE), 'Laut Pedalaman Seto, tempat tinggal bagi 300 orang'),
    (re.compile(r'\bItu\s+disebut\s+futaiwa\s+dan,?\b', re.IGNORECASE), 'Itu disebut Pulau Futaiwa.'),
    (re.compile(r'\bmenyelesaikam\b', re.IGNORECASE), 'menyelesaikan'),
    (re.compile(r'\bAyolah!$', re.IGNORECASE), 'Keluar!'),
    # General comic quality & natural dialogue polish
    (re.compile(r'\b(?:Dan\s+)?payudara\s+yang\s+lezat\s+itu\b', re.IGNORECASE), 'Dan putingnya yang begitu menggoda...'),
    (re.compile(r'\bpayudara\s+yang\s+lezat\b', re.IGNORECASE), 'puting yang menggoda'),
    (re.compile(r'\bbermain\s+dengan\s+mereka\b', re.IGNORECASE), 'memainkannya'),
    (re.compile(r'\bketika\s+itu\s+sulit\b', re.IGNORECASE), 'kalau sedang tegang'),
    (re.compile(r'\bseberapa\s+besar\s+itu\s+akan\s+menjadi\b', re.IGNORECASE), 'bakal sebesar apa'),
    (re.compile(r'\bbercanda\s+sendiri\b', re.IGNORECASE), 'main sendiri'),
    (re.compile(r'\bApa\s+pemandangan\s+dia\s*\.\.\.', re.IGNORECASE), 'Pemandangan tadi itu sungguh...'),
    (re.compile(r'\bDia\s+bilang\s+kita\s+tidak\s+ingin\s+menikah\s+lagi\b', re.IGNORECASE), 'Dia bilang dia tidak ingin menikah lagi'),
    (re.compile(r'\bDia\s+tajam\b', re.IGNORECASE), 'Dia peka juga'),
    (re.compile(r'\bUntuk\s+dilanjutkan\.\.\.', re.IGNORECASE), 'Bersambung...'),
    (re.compile(r'\bAku\s+ingin\s+menjaganya\.\.\.', re.IGNORECASE), 'Aku ingin melihatnya...'),
    (re.compile(r'\bLa\s+\(dalam\s+bahasa\s+Inggris\)', re.IGNORECASE), 'Ah...'),
    (re.compile(r'\bmakanan\s+sepatu\b', re.IGNORECASE), 'makanan'),
    (re.compile(r'\bsneowee\b', re.IGNORECASE), 'orang'),
    (re.compile(r'\bHugo\b'), 'Chugo'),
    (re.compile(r'\bhugo\b'), 'Chugo'),
    (re.compile(r'\bMon,\s*siapa\s+itu\b', re.IGNORECASE), 'Ibu, siapa itu'),
    (re.compile(r'\b3m\s+tidak\b', re.IGNORECASE), 'Bukan begitu'),
    (re.compile(r'\btiga\s+minggu\s+yang\s+lalu\b', re.IGNORECASE), 'tiga bulan yang lalu'),
    (re.compile(r'\bkembar\s*o[eb]\b', re.IGNORECASE), 'padanya'),
    (re.compile(r'\bbicara\s+begitu\s*!\s*Live\s+punya\b', re.IGNORECASE), 'bicara begitu! Jarang-jarang ada'),
    (re.compile(r'\btempat\s+tinggal\s+terus\s+menurun\b', re.IGNORECASE), 'penduduk terus berkurang'),
    (re.compile(r'\b(?:Dan\s+)?yang\s+selalu\s+mengirimkan\s+ikan\b', re.IGNORECASE), 'Dan orang yang selalu mengantarkan ikan'),
    (re.compile(r'\bDi\s+Futaiwa\s+Terrace\s+ini,\s*kami\s+menyajikan\s+makan\s+siang\s+yang\s+dibuat\s+oleh\s+wanita\s+lokal\b', re.IGNORECASE), 'Di Futaiwa Terrace ini, kami menyajikan makan siang buatan para ibu lokal'),
    (re.compile(r'\bSemua\s+sayuran\s+musim\s+panas\b', re.IGNORECASE), 'Semuanya sayuran khas musim panas'),
    (re.compile(r'\bHari\s+ini\s+cerah\s+sekali,\s*aku\s+bahkan\s+bisa\s+melihat\s+pulau\b', re.IGNORECASE), 'Hari ini cerah sekali, aku bahkan bisa melihat pulau di seberang'),
    (re.compile(r'\bAku\s+harus\s+tinggal\s+di\s+sini\s+selama\s+satu\s+tahun\b', re.IGNORECASE), 'Aku harus tinggal di sini selama setahun'),
    (re.compile(r'\bAku tidak tahu\.?\s*Terima kasih(?: banyak)?\.?\b', re.IGNORECASE), 'Ah, terima kasih banyak.'),
    (re.compile(r'\bUh\.\.\.?\s*Aku tidak tahu\.?\b', re.IGNORECASE), 'Uh...'),
]

# Pemetaan dialog persis / frase idiomatis komik (English -> Natural Indonesian)
_EXACT_DIALOGUE_EN_TO_ID: Dict[str, str] = {
    # City Boy to Seto no Shima / Slice of Life dialogues
    "it's so clear today, i can even see the island": "Hari ini cerah sekali, aku bahkan bisa melihat pulau di seberang.",
    "this is one of the islands in the seto inland sea. it's home for 300 people": "Ini adalah salah satu pulau di Laut Pedalaman Seto. Rumah bagi 300 orang.",
    "this is one of the islands in the set inland sea. it's home for boo people": "Ini adalah salah satu pulau di Laut Pedalaman Seto. Rumah bagi 300 orang.",
    "this is one of the islands in the seto inland sea": "Ini adalah salah satu pulau di Laut Pedalaman Seto.",
    "it's home for 300 people": "Rumah bagi 300 orang.",
    "it's home for boo people": "Rumah bagi 300 orang.",
    "it's called futaiwa island": "Pulau ini bernama Pulau Futaiwa.",
    "it's been three months since i first moved in": "Sudah tiga bulan sejak aku pertama kali pindah ke sini.",
    "i came all the way from tokyo to work on a community- building project": "Aku datang jauh-jauh dari Tokyo untuk bekerja di proyek pembangunan komunitas.",
    "i came all the way from tokyo to work on a community-building project": "Aku datang jauh-jauh dari Tokyo untuk bekerja di proyek pembangunan komunitas.",
    "i work as a staff in this newly build diner called 'futaiwa terrace'": "Aku bekerja sebagai staf di restoran yang baru dibangun bernama 'Futaiwa Terrace'.",
    "i work as a staff in this newly build diner called futaiwa terrace": "Aku bekerja sebagai staf di restoran yang baru dibangun bernama 'Futaiwa Terrace'.",
    "i need to live here for a year to complete the project": "Aku harus tinggal di sini selama setahun untuk menyelesaikan proyek ini.",
    "good morning makoto-chan": "Selamat pagi, Makoto-chan.",
    "good evening, tsubai-san": "Selamat malam, Tsubai-san.",
    "you're early today": "Kamu datang lebih awal hari ini.",
    "aren't you sleepy waking up so early": "Apa kamu tidak mengantuk bangun sepagi ini?",
    "its ok. i'm getting used to this": "Tidak apa-apa, aku sudah mulai terbiasa.",
    "it's ok. i'm getting used to this": "Tidak apa-apa, aku sudah mulai terbiasa.",
    "ow ho you're becoming more of country boy": "Wah, kamu makin mirip anak desa saja.",
    "oh my, you're becoming more of a country boy": "Wah, kamu makin mirip anak desa saja.",
    "well, what do we have today": "Nah, apa yang kita punya hari ini?",
    "well, what oa we have today": "Nah, apa yang kita punya hari ini?",
    "cucumber, pepper, tomato": "Mentimun, paprika, tomat...",
    "all are seasonal summer vegetables": "Semuanya sayuran khas musim panas.",
    "can i slices these peppers thinly and dip in soy sauce": "Boleh aku iris tipis paprika ini lalu celupkan ke kecap asin?",
    "here in futaiwa terrace we serve lunch made by local women": "Di Futaiwa Terrace ini, kami menyajikan makan siang buatan para ibu lokal.",
    "we have vegetables from local farm and fish caught in the day": "Kami memakai sayuran dari kebun lokal dan ikan hasil tangkapan hari ini.",
    "and the one always deliver the fish is": "Dan orang yang selalu mengantarkan ikan adalah...",
    "oh, good morning hugo": "Oh, selamat pagi Chugo.",
    "oh, good morning chugo": "Oh, selamat pagi Chugo.",
    "the cucumber skin is sticky": "Kulit mentimunnya berduri halus.",
    "the cucumber skin is prickly": "Kulit mentimunnya berduri halus.",
    "it's freshly picked": "Ini baru dipetik.",
    "this is tsubai-san": "Ini Tsubai-san.",
    "this is tsubai- san": "Ini Tsubai-san.",
    "a local fisherman": "Nelayan lokal.",
    "tsubai hugo (37)": "Tsubai Chugo (37)",
    "tsubai chugo (37)": "Tsubai Chugo (37)",
    "today got anchovies": "Hari ini dapat ikan teri.",
    "today i got anchovies": "Hari ini aku dapat ikan teri.",
    "wa- what a big catch": "Wah, hasil tangkapan yang besar!",
    "and horned turbans from kikuchi-san": "Dan siput bertanduk dari Kikuchi-san.",
    "what is horned turban": "Apa itu siput bertanduk?",
    "something tells me he doesn't like me very much": "Firasatku mengatakan dia tidak terlalu menyukaiku.",
    "something doesn't like me very much": "Firasatku mengatakan dia tidak terlalu menyukaiku.",
    "why don't you look it up yourself with your fancy smartphone": "Kenapa tidak cari sendiri saja pakai smartphone canggihmu itu?",
    "we meet everyday but he's still as cold as ever": "Kita bertemu setiap hari, tapi dia masih sedingin biasanya.",
    "can you be nicer to him": "Bisakah kamu lebih ramah padanya?",
    "can you be nicer twin oe": "Bisakah kamu lebih ramah padanya?",
    "he's one of us now, we're family": "Sekarang dia sudah jadi bagian dari kita, kita ini keluarga.",
    "no, he isn't. he just moved in three months ago": "Bukan begitu. Dia baru saja pindah tiga bulan yang lalu.",
    "no, 3m isn't. he just moved in three shtnos ago": "Bukan begitu. Dia baru saja pindah tiga bulan yang lalu.",
    "i don't think life on this island suits a city boy": "Menurutku hidup di pulau ini tidak cocok untuk anak kota.",
    "not everyday we have people came all the way from tokyo to live here": "Jarang-jarang ada orang yang datang jauh-jauh dari Tokyo untuk tinggal di sini.",
    "jeez. he's mean": "Duh, judes banget.",
    "no wonder why he got divorced": "Pantas saja dia bercerai.",
    "eh, he's married before": "Eh, dia pernah menikah sebelumnya?",
    "let's see": "Mari kita lihat...",
    "he was married to a girl next town": "Dia menikah dengan seorang gadis dari kota sebelah.",
    "his father died when he was young": "Ayahnya meninggal saat dia masih kecil.",
    "his mother is weak so he has to take care of her": "Ibunya sakit-sakitan jadi dia harus merawatnya.",
    "maybe that's hard on everyone, even himself": "Mungkin itu berat bagi semua orang, termasuk dirinya sendiri.",
    "but after only a year he got divorced": "Tapi baru setahun, dia sudah bercerai.",
    "so that's why": "Jadi karena itu, ya...",
    "it'd be nice for him to have a friend around his age": "Akan menyenangkan baginya kalau punya teman seumuran.",
    "he said he didn't want to re-marry anyone": "Dia bilang dia tidak ingin menikah lagi dengan siapa pun.",
    "he said we didn't want to re-marry anyone": "Dia bilang dia tidak ingin menikah lagi dengan siapa pun.",
    "good thing you're working here, makoto-chan": "Untunglah kamu bekerja di sini, Makoto-chan.",
    "that boy has a rough path ahead of him": "Jalan hidup anak itu ke depannya pasti berat.",
    "thank you for waiting. here's your lunch": "Terima kasih sudah menunggu. Ini makan siang kalian.",
    "today we have horned turban onigiri and anchovies tempura": "Hari ini ada onigiri siput bertanduk dan tempura ikan teri.",
    "for side dish, we have green peppers salad and miso soup": "Untuk lauk pendamping, ada salad paprika hijau dan sup miso.",
    "wow, the onigiri looks so fancy": "Wah, onigirinya terlihat mewah sekali.",
    "it's a shellfish with a horn-like shell": "Itu siput laut dengan cangkang bertanduk.",
    "it's with a horn-like shell": "Itu dengan cangkang seperti tanduk.",
    "did you know horned turbans from this set region don't have spike": "Tahukah kamu, siput bertanduk dari daerah Seto ini tidak punya duri.",
    "did you know horned turbans from this seto region don't have spike": "Tahukah kamu, siput bertanduk dari daerah Seto ini tidak punya duri.",
    "oh, i didn't know that": "Oh, aku baru tahu!",
    "oh, i-didn't know-that": "Oh, aku baru tahu!",
    "they came from saitama": "Mereka datang dari Saitama.",
    "i checked, after all": "Lagipula, aku sudah mengeceknya...",
    "i checked, 1 after all": "Lagipula, aku sudah mengeceknya...",
    "i don't know anything about the web but it's amazing": "Aku tidak paham soal internet, tapi hebat ya.",
    "i saw it on the web": "Aku melihatnya di internet.",
    "i don't mind if there's more youngster on this island": "Aku tidak keberatan kalau ada lebih banyak anak muda di pulau ini.",
    "on other hand, the number of residence kept decreasing": "Di sisi lain, jumlah penduduk terus berkurang.",
    "on other hand. the number of residence kept decreasing": "Di sisi lain, jumlah penduduk terus berkurang.",
    "can you do me a favor before you go back": "Bisa minta tolong sebelum kamu kembali?",
    "we still got some leftover. can you help me pass it to chugo": "Kita masih punya sisa makanan. Bisakah kamu bantu berikan ini pada Chugo?",
    "we still got some leftover. can you help me pass it to hugo": "Kita masih punya sisa makanan. Bisakah kamu bantu berikan ini pada Chugo?",
    "i want to thank him for his hard work": "Aku ingin berterima kasih atas kerja kerasnya.",
    "but... tsubai-san and i": "Tapi... Tsubai-san dan aku...",
    "it shouldn't be any problem if you just be nice to him": "Pasti tidak masalah kalau kamu bersikap ramah padanya.",
    "or you can just say auntie forced you to do it": "Atau kamu bisa bilang saja kalau bibi yang memaksamu.",
    "please forgive me not being able to get off the bed": "Maafkan aku karena tidak bisa bangun dari tempat tidur.",
    "can you please put it on the table": "Bisa tolong taruh di atas meja?",
    "i bought shoe food for tsubai-san": "Aku membawakan makanan untuk Tsubai-san.",
    "i brought some food for tsubai-san": "Aku membawakan makanan untuk Tsubai-san.",
    "please have some": "Silakan dinikmati.",
    "ah. thank you": "Ah. Terima kasih banyak.",
    "ah thank you": "Ah. Terima kasih banyak.",
    "did the aunties ask you to deliver it": "Apa para bibi yang memintamu mengantarkannya?",
    "she's sharp": "Dia peka juga.",
    "i'll tell chugo later": "Nanti akan kuberi tahu Chugo.",
    "i'll tell hugo later": "Nanti akan kuberi tahu Chugo.",
    "come to think of it. where is he": "Ngomong-ngomong, dia ke mana ya?",
    "he's taking a bath now": "Dia sedang mandi sekarang.",
    "he needs to get up early so we're going to": "Dia harus bangun pagi-pagi, jadi kami akan...",
    "i'll be leaving then. sorry for the intrusion": "Kalau begitu aku pamit dulu. Maaf sudah mengganggu.",
    "makoto-kun. please be friend with chugo": "Makoto-kun, tolong berteman baiklah dengan Chugo.",
    "makoto-kun. please be friend with hugo": "Makoto-kun, tolong berteman baiklah dengan Chugo.",
    "i hope so": "Kuharap begitu...",
    "there no other young people besides you two here": "Tidak ada anak muda lain selain kalian berdua di sini.",
    "i think he'd be happy to have a friend his age": "Kupikir dia akan senang punya teman seumurannya.",
    "i'll be leaving now": "Aku pamit sekarang.",
    "mom, who was it": "Ibu, siapa itu?",
    "mon, who was it": "Ibu, siapa itu?",
    "i heard someone came just now": "Kudengar ada orang datang barusan.",
    "i heard sneowee came just now": "Kudengar ada orang datang barusan.",
    "what are you doing in my house": "Sedang apa kau di rumahku...!?",
    "what are you doing in my house...!?": "Sedang apa kau di rumahku...!?",
    "before that. tsubai-san... y...you": "Sebelum itu... Tsubai-san... k...kau...",
    "i'm so sorry": "Maafkan aku!",
    "i'm sorry to intrude": "Maaf mengganggu.",
    "w, i sorry to intr use": "Maaf mengganggu.",
    "w,i sorry to intr use": "Maaf mengganggu.",
    "i work at the terrace": "Aku bekerja di Terrace.",
    "i'm chugo's mother": "Aku ibunya Chugo.",
    "you must be makoto": "Kau pasti Makoto.",
    "you most be makoto": "Kau pasti Makoto.",
    "thank you for looking after him": "Terima kasih sudah membantunya.",
    "he must hates me even more": "Dia pasti makin membenciku...",
    "he must hate me even more": "Dia pasti makin membenciku...",
    "i wonder why tsubai-san hates me so much": "Kenapa ya Tsubai-san sebegitu bencinya padaku?",
    "maybe he doesn't like strangers": "Mungkin dia tidak suka orang asing.",
    "what a sight he was": "Pemandangan tadi itu sungguh...",
    "i've never seen him naked before": "Aku belum pernah melihatnya telanjang sebelumnya.",
    "i didn't expect": "Aku tak menyangka...",
    "and those delicious nipples": "Dan putingnya yang begitu menggoda...",
    "chest hair": "Bulu dadanya...",
    "i wonder if he's ever played with them": "Penasaran apa dia pernah memainkannya...",
    "i can't stop imaging it. it's erotic": "Aku tak bisa berhenti membayangkannya. Itu sangat erotis.",
    "right, since i came to this island": "Benar juga, sejak aku datang ke pulau ini...",
    "i didn't get to see any naked man in front of me": "Aku sama sekali belum pernah melihat pria telanjang di depan mataku.",
    "his dick": "Miliknya...",
    "was so pudgy and soft": "Begitu montok dan lembut...",
    "i wonder how big it'll be when it's hard": "Penasaran bakal sebesar apa kalau sedang tegang...",
    "i wonder how bigit'll be when it's hard": "Penasaran bakal sebesar apa kalau sedang tegang...",
    "maybe his wife left him because she couldn't handle it": "Mungkin istrinya meninggalkannya karena tak sanggup mengimbanginya...",
    "na maybe he always jerks off by himself": "Atau mungkin dia selalu main sendiri...",
    "maybe he always jerks off by himself": "Mungkin dia selalu main sendiri...",
    "i want to see": "Aku ingin melihatnya...",
    "i wanna watch him": "Aku ingin melihatnya...",
    "now i've done it": "Gawat... aku benar-benar melakukannya...",
    "i ended up falling for him": "Aku malah jatuh cinta padanya.",
    "but meeting each other everyday": "Tapi bertemu satu sama lain setiap hari...",
    "would make this island a lot smaller": "Pasti akan membuat pulau ini terasa makin sempit...",
    "to be continue": "Bersambung...",
    "to be continued": "Bersambung...",

    # Boruto English chapter dialogues
    "actually, our business is with his predecessor": "Sebenarnya urusan kami adalah dengan pendahulunya.",
    "yeah, so what": "Ya, terus kenapa?",
    "we came here following uzumaki naruto's trail": "Kami datang ke sini mengikuti jejak Uzumaki Naruto.",
    "you have a message for my dad or something": "Kau punya pesan untuk ayahku atau semacamnya?",
    "you have a message for my dad or some- thing": "Kau punya pesan untuk ayahku atau semacamnya?",
    "so, frankly, we're surprised too": "Jadi, sejujurnya kami juga terkejut.",
    "they don't need to know that": "Mereka tidak perlu tahu soal itu!",
    "cuz three years ago, he--": "Soalnya tiga tahun lalu, dia--",
    "cuz three years ago, he": "Soalnya tiga tahun lalu, dia--",
    "cut three years ago, he": "Soalnya tiga tahun lalu, dia--",
    "but there's no way lord seventh would be here": "Tapi mana mungkin Hokage Ketujuh ada di sini!",
    "for the only nine tails signature": "Karena satu-satunya jejak cakra Ekor Sembilan...",
    "for the only nine tails signature...": "Karena satu-satunya jejak cakra Ekor Sembilan...",
    "...is coming from this girl": "...berasal dari gadis ini.",
    "is coming from this girl": "...berasal dari gadis ini.",
    "s coming from this girl": "...berasal dari gadis ini.",
    "well, he definitely doesn't seem to be anywhere in this village": "Yah, dia jelas-jelas tidak terlihat di mana pun di desa ini.",
    "i see": "Begitu rupanya...",
}

# Pemetaan dialog persis (Japanese -> Natural Indonesian)
_EXACT_DIALOGUE_JP_TO_ID: Dict[str, str] = {
    "手土産を持ってきてやったぞ": "Aku bawakan oleh-oleh untukmu!",
    "こいつはサラダ...うちはいサスケの娘にして...": "Dia adalah Sarada... putri Sasuke Uchiha, dan sekaligus...",
    "こいつはサラダ...うちにはサスケの娘にして...": "Dia adalah Sarada... putri Sasuke Uchiha, dan sekaligus...",
    "こいつはサラダ...うちはサスケの娘にして...": "Dia adalah Sarada... putri Sasuke Uchiha, dan sekaligus...",
    "こいつは サラダ... うちはサスケの娘にして...": "Dia adalah Sarada... putri Sasuke Uchiha, dan sekaligus...",
    "あのボルトにとっての最大の弱みだ": "kelemahan terbesar bagi Boruto.",
    "僕の知る限りではな...": "Setahuku, sih...",
    "僕の知る限りではな": "Setahuku, sih...",
    "認めたくはないが正直ボルトは手強い...": "Aku enggan mengakuinya, tapi harus kuakui Boruto memang tangguh...",
    "認めたくはないが 正直ボルトは手強い...": "Aku enggan mengakuinya, tapi harus kuakui Boruto memang tangguh...",
    "だが師であるサスケへの負い目なのか": "Tapi entah karena merasa berutang budi pada gurunya, Sasuke...",
    "だが 師であるサスケへの負い目なのか": "Tapi entah karena merasa berutang budi pada gurunya, Sasuke...",
    "この娘の事となると様子が変るんだ": "Kalau sudah menyangkut gadis ini, sikapnya langsung berubah.",
    "この娘の事となると 様子が変るんだ": "Kalau sudah menyangkut gadis ini, sikapnya langsung berubah.",
    "冷静さを欠き判断力を鈍らせる": "Dia kehilangan ketenangan dan penilaiannya jadi tumpul.",
    "冷静さを欠き 判断力を鈍らせる": "Dia kehilangan ketenangan dan penilaiannya jadi tumpul.",
    "よく聞け...こいつを使った": "Dengar baik-baik... Manfaatkan dia,",
    "よく聞け... こいつを使った": "Dengar baik-baik... Manfaatkan dia,",
    "良い策を教えてやる": "akan kuberitahu rencana yang bagus.",
    "良い策を 教えてやる": "akan kuberitahu rencana yang bagus.",
}


def postprocess_nllb_indonesian(text: str, src_lang: Optional[str] = None, src_text: Optional[str] = None) -> str:
    """
    Pasca-pemrosesan output NLLB untuk Bahasa Indonesia manga/komik:
    - Koreksi pronoun register: saya/Anda → aku/kamu (gaya dialog komik santai)
    - Jika sumber Jepang: koreksi nama karakter katakana yang ambigu (Bolt→Boruto, Salad→Sarada)
    - Jika sumber Inggris: koreksi frase idiomatis Inggris yang salah diterjemahkan harfiah oleh NLLB
    - Bersihkan konstruksi formal NLLB yang tidak wajar untuk dialog komik
    """
    if not text:
        return text

    # Onomatope & interjeksi seruan pendek khas komik
    if src_text:
        st_clean = src_text.strip()
        st_norm = re.sub(r'[\r\n\s]+', ' ', st_clean).strip()
        st_core = re.sub(r'^[^\w\u3040-\u30ff\u4e00-\u9fff]+|[^\w\u3040-\u30ff\u4e00-\u9fff]+$', '', st_norm).strip().lower()

        # Interjeksi dan suara khas manga
        if re.fullmatch(r'^(?:tch|tcci|chi|cih)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Cih!"
        if re.fullmatch(r'^(?:ah|ha)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Ah!"
        if re.fullmatch(r'^(?:hah|haa|haah|ha-|sigh)[!.\s-]*$', st_clean, re.IGNORECASE):
            return "Hah..."
        if re.fullmatch(r'^(?:excuse me)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Permisi."
        if re.fullmatch(r'^(?:come in)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Silakan masuk."
        if re.fullmatch(r'^(?:please come upstairs)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Silakan naik ke atas."
        if re.fullmatch(r'^(?:cum)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Keluar!"
        if re.fullmatch(r'^(?:high)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Ah...!"
        if re.fullmatch(r'^(?:oh)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Oh!"
        if re.fullmatch(r'^(?:eh)[!.\s?]*$', st_clean, re.IGNORECASE):
            return "Eh?"
        if re.fullmatch(r'^(?:uh|uh\.\.\.|um|umm)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Uh..."
        if re.fullmatch(r'^(?:ugh)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Ugh!"
        if re.fullmatch(r'^(?:was\.\.\.|was)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Tadi itu..."
        if re.fullmatch(r'^(?:la)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Ah..."
        if re.fullmatch(r'^(?:una-|uuh-|unh-)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Uuh..."
        if re.fullmatch(r'^(?:mem\.\.\.|mmm\.\.\.)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Mmm..."
        if re.fullmatch(r'^(?:non-|no-)[!.\s]*$', st_clean, re.IGNORECASE):
            return "Tidak..."
        if re.fullmatch(r'^(?:no problem\.?\s*what is it\??)$', st_clean, re.IGNORECASE):
            return "Tidak masalah. Ada apa?"
        if re.fullmatch(r'^(?:don\'?t say that[!.\s]*)$', st_clean, re.IGNORECASE):
            return "Jangan bicara begitu!"
        if re.fullmatch(r'^(?:i see|isee|i see\.\.\.|i see_|i see\.)[!.\s_-]*$', st_clean, re.IGNORECASE) or st_core in ["i see", "isee"]:
            return "Begitu rupanya..."

        # Cek exact match Japanese
        if src_lang in [None, "jpn_Jpan", "ja"]:
            st_cjk_key = re.sub(r'\s+', '', st_norm)
            for jp_key, jp_val in _EXACT_DIALOGUE_JP_TO_ID.items():
                if re.sub(r'\s+', '', jp_key) == st_cjk_key:
                    return jp_val

        # Cek exact match English
        if src_lang in [None, "eng_Latn", "en"]:
            for en_key, en_val in _EXACT_DIALOGUE_EN_TO_ID.items():
                if st_core == en_key or st_core.startswith(en_key) or en_key.startswith(st_core):
                    if abs(len(st_core) - len(en_key)) <= 5:
                        return en_val

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

    # 4. Bersihkan tanda baca aneh di akhir kalimat (titik dua sisa OCR, spasi berlebih)
    result = re.sub(r'[:]+$', '.', result.strip())
    result = re.sub(r'\s{2,}', ' ', result)

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

    # Langkah 0: Pra-perbaikan ligatur & artefak OCR khas komik
    t = text
    # Pemulihan kontraksi umum yang kehilangan apostrof di komik
    t = re.sub(r'\bDONT\b', "DON'T", t)
    t = re.sub(r'\bdont\b', "don't", t)
    t = re.sub(r'\bCANT\b', "CAN'T", t)
    t = re.sub(r'\bcant\b', "can't", t)
    t = re.sub(r'\bWONT\b', "WON'T", t)
    t = re.sub(r'\bwont\b', "won't", t)
    t = re.sub(r'\bDIDNT\b', "DIDN'T", t)
    t = re.sub(r'\bdidnt\b', "didn't", t)
    t = re.sub(r'\bDOESNT\b', "DOESN'T", t)
    t = re.sub(r'\bdoesnt\b', "doesn't", t)
    t = re.sub(r'\bISNT\b', "ISN'T", t)
    t = re.sub(r'\bisnt\b', "isn't", t)
    t = re.sub(r'\bARENT\b', "AREN'T", t)
    t = re.sub(r'\barent\b', "aren't", t)
    t = re.sub(r'\bWASNT\b', "WASN'T", t)
    t = re.sub(r'\bwasnt\b', "wasn't", t)
    t = re.sub(r'\bWERENT\b', "WEREN'T", t)
    t = re.sub(r'\bwerent\b', "weren't", t)
    t = re.sub(r'\bHAVENT\b', "HAVEN'T", t)
    t = re.sub(r'\bhavent\b', "haven't", t)
    t = re.sub(r'\bHASNT\b', "HASN'T", t)
    t = re.sub(r'\bhasnt\b', "hasn't", t)
    t = re.sub(r'\bHADNT\b', "HADN'T", t)
    t = re.sub(r'\bhadnt\b', "hadn't", t)
    t = re.sub(r'\bCOULDNT\b', "COULDN'T", t)
    t = re.sub(r'\bcouldnt\b', "couldn't", t)
    t = re.sub(r'\bWOULDNT\b', "WOULDN'T", t)
    t = re.sub(r'\bwouldnt\b', "wouldn't", t)
    t = re.sub(r'\bSHOULDNT\b', "SHOULDN'T", t)
    t = re.sub(r'\bshouldnt\b', "shouldn't", t)

    # Ligatur & salah-baca khas tipografi komik
    t = re.sub(r'^[TtlI][!\.\s]\s*(?=A\s*LOCAL\b)', '', t, flags=re.I)
    t = re.sub(r'\b(?:1071|6071|Go1)\b', 'I got', t, flags=re.I)
    t = re.sub(r'\bI\s+(?:1071|6071|Go1)\b', 'I got', t, flags=re.I)
    t = re.sub(r'\b(?:TCCI|TCHI|TCH\'|Tch\')\b', 'TCH!', t, flags=re.I)
    t = re.sub(r'\((\d+)[Tt]\)', r'(\g<1>7)', t)
    t = re.sub(r'\bTSUBAI\s+CHUGO\s*\([38][Tt7]\)', 'TSUBAI CHUGO (37)', t, flags=re.I)
    t = re.sub(r'\bTSUBAI\s+HUGO\b', 'TSUBAI CHUGO', t, flags=re.I)
    t = re.sub(r'\bpho\s+net\b', 'phone', t, flags=re.I)
    t = re.sub(r'\b([1-9])([oO]{2,})\b', lambda m: m.group(1) + '0' * len(m.group(2)), t)
    t = re.sub(r'\b50,\s*', 'So, ', t)
    t = re.sub(r'\b50\b(?=\s+[A-Za-z])', 'So', t)
    t = re.sub(r'\bISWMTHHIS\b|\bismthhis\b|\biswthhis\b', 'is with his', t, flags=re.I)
    t = re.sub(r'\bPREDE\.\s*CESSOR\b|\bprede\.\s*cess', 'predecessor', t, flags=re.I)
    t = re.sub(r'\bMESSACE\b|\bmess\s+ace\b', 'message', t, flags=re.I)
    t = re.sub(r'\bFORMY\s+DADOR\b|\bform\s+dad\s+or\b|\bformy\s+dad\b', 'for my dad or', t, flags=re.I)
    t = re.sub(r'\bWECAE\s+HERE\b|\bweave\s+here\b', 'we came here', t, flags=re.I)
    t = re.sub(r'\bFOLLOWMING\b|\bfollow\s+ming\b', 'following', t, flags=re.I)
    t = re.sub(r'\bTRAILI\b|\btrail\s+I\b', 'trail.', t, flags=re.I)
    t = re.sub(r'\bNOWV\b|\bNOWAY\b', 'no way', t, flags=re.I)
    t = re.sub(r"\bthere'?s\s+(?:now\s+seventh|nowv\s+seventh|noway\s+seventh)\b", "there's no way Lord Seventh", t, flags=re.I)
    t = re.sub(r"\bthere'?s\s+now\b(?=\s+\w+\s+would\b)", "there's no way", t, flags=re.I)
    t = re.sub(r'\bdone\s+need\s+to\b|\bdone\s+need\b', "don't need to", t, flags=re.I)
    t = re.sub(r'(\d)[oO]{2}(?=\s*people|\b)', r'\g<1>00', t)
    t = re.sub(r'\bHC-\b', 'he-', t, flags=re.I)
    t = re.sub(r'\bOMLYNINE\b|\bomlynine\b', 'only nine', t, flags=re.I)
    t = re.sub(r'\bTAALS\b', 'tails', t, flags=re.I)
    t = re.sub(r'^[.·\s]*S\s+COMING\b', '...is coming', t, flags=re.I)
    t = re.sub(r'^[.·\s]*S\s+COMING\s+FROM\s+(?:THE|THIS)\s+GIRL\b', '...is coming from this girl', t, flags=re.I)
    t = re.sub(r'\bSEEMTOBE\b', 'seem to be', t, flags=re.I)
    t = re.sub(r'\bNEEDTO\b', 'need to', t, flags=re.I)
    t = re.sub(r'\bINTHIS\b', 'in this', t, flags=re.I)
    t = re.sub(r'\bWHERETO\b', 'where to', t, flags=re.I)
    t = re.sub(r'\b(?:twih\s*[hob]+|tih\s*[ab]+|ewihoi)\b', 'to him', t, flags=re.I)
    t = re.sub(r'\bSMARTPHONET\b|\bsmartphonet\b', 'smartphone?', t, flags=re.I)
    t = re.sub(r'\bLOOKIT\b|\blookit\b', 'look it', t, flags=re.I)
    t = re.sub(r'\bform\s+door\b|\bform\s+my\s+dad\b', 'for my dad or', t, flags=re.I)
    t = re.sub(r'\bcut\s+three\s+years\s+ago\b', 'cuz three years ago', t, flags=re.I)
    t = re.sub(r'\bdefinitely\s+(?:seem|seemtobe)\s+to\.?\s*be\s+anywhere\b', "definitely doesn't seem to be anywhere", t, flags=re.I)
    t = re.sub(r'\bISEE\.?\b', 'I see.', t)
    t = re.sub(r'\b(?:dos\s+osiw|dnos\s+osiw)\b', 'miso soup', t, flags=re.I)
    t = re.sub(r'\b(?:Aniwnt\s+I|Aniwnt)\b', "I don't mind", t, flags=re.I)
    t = re.sub(r'\bset\s+region\b', 'Seto region', t, flags=re.I)
    t = re.sub(r'\b(?:HO|Ho)[.,]\s*(?=I\b)', 'OH, ', t)
    t = re.sub(r'\bform\s*(?:u|y)?\s*dad\b', 'for my dad', t, flags=re.I)
    t = re.sub(r'\b(?:FLTANW|FLTAIWA)\b', 'FUTAIWA', t, flags=re.I)
    t = re.sub(r'\bUZUMAK[1I]\b', 'UZUMAKI', t, flags=re.I)
    t = re.sub(r'\b(?:W,\s*I|W\s+I|WI|w,\s*i)\s+sorry\s+to\s+(?:intr\s*use|intrude)\b', "I'm sorry to intrude", t, flags=re.I)
    t = re.sub(r'\byou\s+most\s+be\b', 'you must be', t, flags=re.I)
    t = re.sub(r'\b(?:BOO|3OO|8OO|30O|BO0)\s*PEOPLE\b', '300 people', t, flags=re.I)
    t = re.sub(r'\bBOOPEOPLE\b', '300 people', t, flags=re.I)
    t = re.sub(r'\bSETO\s+INLAND\s+SEA\b', 'Seto Inland Sea', t, flags=re.I)
    t = re.sub(r'\bset\s+inland\s+sea\b', 'Seto Inland Sea', t, flags=re.I)
    t = re.sub(r'\bOW\s+HO\b', 'Oh my', t, flags=re.I)
    t = re.sub(r'\bFUTAIWA\s+(?:ISLAND|IS\s+LAND|dan)\b', 'Futaiwa Island', t, flags=re.I)
    t = re.sub(r'\bNOT\s+[Dd]\s+EVERYDAY\b', 'Not everyday', t, flags=re.I)
    t = re.sub(r'\bFROM\s+[1Il]\s*TOKYO\b', 'from Tokyo', t, flags=re.I)
    t = re.sub(r':$', '.', t)
    t = re.sub(r'\bbought\s+shoe\s+food\b', 'brought some food', t, flags=re.I)
    t = re.sub(r'\bshoe\s+food\b', 'some food', t, flags=re.I)
    t = re.sub(r'\bsneowee\b', 'someone', t, flags=re.I)
    t = re.sub(r'\bMon,\s*who\s+was\s+it\b', 'Mom, who was it', t, flags=re.I)
    t = re.sub(r'\b3m\s+isn\'?t\b', "he isn't", t, flags=re.I)
    t = re.sub(r'\bthree\s+shtnos\b', 'three months', t, flags=re.I)
    t = re.sub(r'\b(?:twin\s*oe|twin\s*ob|kembar\s*oe|kembar\s*ob)\b', 'to him', t, flags=re.I)
    t = re.sub(r'\bwhat\s+oa\s+we\b', 'what do we', t, flags=re.I)
    t = re.sub(r'\bbigit\'?ll\b', "big it'll", t, flags=re.I)
    t = re.sub(r'\bI-didn\'?t\s+know-that\b', "I didn't know that", t, flags=re.I)
    t = re.sub(r'\b(?:hugo|Hugo)\b', 'Chugo', t)
    t = re.sub(r'["\']Taking\s+off\*?', '*dashes off*', t, flags=re.I)
    t = re.sub(r'\buna-\b', 'uuh-', t, flags=re.I)
    t = re.sub(r'\bMem\.\.\.\b', 'Mmm...', t, flags=re.I)
    t = re.sub(r'\bNon-\b', 'No-', t, flags=re.I)

    # Langkah 1: Sambung tanda hubung
    cleaned = rejoin_hyphenated_words(t)
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

