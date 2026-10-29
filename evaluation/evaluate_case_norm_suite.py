import sys, os, json, re
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from src.translator import MangaTranslator
from src.text_utils import normalize_latin_case

with open("evaluation/ocr_latin_ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

translator = MangaTranslator(device="cuda")

# Common English words that shouldn't leak into Indonesian translations
ENGLISH_STOPWORDS = {"the", "and", "that", "this", "what", "with", "from", "they", "there", "about", "thing", "things"}

negation_markers_src = ["don't", "can't", "not", "without", "too bad", "no way", "isn't"]
negation_markers_id = ["tidak", "tak", "bukan", "jangan", "tanpa", "sayang", "belum"]

before_results = []
after_results = []

raw_texts = list(gt.values())
norm_texts = [normalize_latin_case(t) for t in raw_texts]

trans_before = translator.translate_batch(raw_texts, src_lang="eng_Latn", tgt_lang="ind_Latn")
trans_after = translator.translate_batch(norm_texts, src_lang="eng_Latn", tgt_lang="ind_Latn")

def count_english_leakage(text):
    words = [w.lower() for w in re.findall(r'\b[A-Za-z]+\b', text)]
    return sum(1 for w in words if w in ENGLISH_STOPWORDS)

leak_before = sum(count_english_leakage(t) for t in trans_before)
leak_after = sum(count_english_leakage(t) for t in trans_after)

neg_preserved_before = 0
neg_preserved_after = 0
total_neg_sources = 0

for src, b_out, a_out in zip(raw_texts, trans_before, trans_after):
    src_has_neg = any(m in src.lower() for m in negation_markers_src)
    if src_has_neg:
        total_neg_sources += 1
        if any(m in b_out.lower() for m in negation_markers_id):
            neg_preserved_before += 1
        if any(m in a_out.lower() for m in negation_markers_id):
            neg_preserved_after += 1

print("\n" + "="*60)
print("HASIL EVALUASI NORMALISASI HURUF KAPITAL LATIN")
print("="*60)
print(f"Total Balon Teruji           : {len(raw_texts)}")
print(f"Total Balon dengan Negasi   : {total_neg_sources}")
print(f"Negasi Terjaga (Before)     : {neg_preserved_before}/{total_neg_sources} ({neg_preserved_before/total_neg_sources*100:.1f}%)")
print(f"Negasi Terjaga (After)      : {neg_preserved_after}/{total_neg_sources} ({neg_preserved_after/total_neg_sources*100:.1f}%)")
print(f"Kebocoran Kata Inggris (Before): {leak_before} kata")
print(f"Kebocoran Kata Inggris (After) : {leak_after} kata")
print("="*60 + "\n")
