import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator

translator = MangaTranslator()
t = translator.translate_batch([
    "This is one of the islands in the Seto inland sea. It's home for 300 people.",
    "It's home for 300 people.",
    "It's home for 3oopeople.",
    "It's home for 300 people"
], src_lang="eng_Latn", tgt_lang="ind_Latn")
for s, r in zip([
    "This is one of the islands in the Seto inland sea. It's home for 300 people.",
    "It's home for 300 people.",
    "It's home for 3oopeople.",
    "It's home for 300 people"
], t):
    print(f"{repr(s)} -> {repr(r)}")
