import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator

translator = MangaTranslator()
t = translator.translate_batch(["Today I got anchovies", "And horned turbans from Kikuchi-san.", "What is horned turban?", "TCH!"], src_lang="eng_Latn", tgt_lang="ind_Latn")
print("Translations:")
for orig, res in zip(["Today I got anchovies", "And horned turbans from Kikuchi-san.", "What is horned turban?", "TCH!"], t):
    print(f"  {repr(orig)} -> {repr(res)}")
