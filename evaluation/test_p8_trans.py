import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator

tr = MangaTranslator()
src = "I don't know anything about the web but it's amazing."
res = tr.translate(src, src_lang="eng_Latn", tgt_lang="ind_Latn")
print("Direct translate:", repr(res))

batch_src = [
    "Thank you for waiting. Here's your lunch.",
    "Today we have horned turban onigiri and anchovies tempura.",
    "For side dish, we have green peppers salad and miso soup",
    "Wow, the onigiri looks so fancy.",
    "It's with a horn-like shell.",
    "Did you know horned turbans from this set region don't have spike.",
    "What is horned turban?",
    "They came from saitama.",
    "I checked, after all...",
    "I don't know anything about the web but it's amazing.",
    "I saw it on the web."
]
batch_res, _ = tr.translate_page_bubbles(batch_src, src_lang="eng_Latn", tgt_lang="ind_Latn")
for s, r in zip(batch_src, batch_res):
    print(f"SRC: {s}\nTGT: {r}\n")
