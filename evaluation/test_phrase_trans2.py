import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator
tr = MangaTranslator()
t = tr.translate("Why don't you look it up yourself with your fancy smartphone?", src_lang="eng_Latn", tgt_lang="ind_Latn")
print("RESULT:", t)
