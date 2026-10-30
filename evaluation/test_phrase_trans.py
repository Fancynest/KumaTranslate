import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator
tr = MangaTranslator()
t = tr.translate("Something tells me he doesn't like me very much.", src_lang="eng_Latn", tgt_lang="ind_Latn")
print("RESULT:", t)
