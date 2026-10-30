import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator
from src.text_utils import postprocess_nllb_indonesian

t = MangaTranslator()

lines = [
    "Actually, our business is with his predecessor.",
    "Yeah, so what?",
    "You have a message for my dad or something?",
    "We came here following Uzumaki Naruto's trail.",
    "So, frankly, we're surprised too.",
    "But there's no way Lord Seventh would be here!",
    "Hey!",
    "Cuz three years ago, he-",
    "They don't need to know that!",
    "I see.",
    "Well, he definitely doesn't seem to be anywhere in this village.",
    "For the only Nine Tails signature...",
    "...is coming from this girl."
]

res = t.translate_batch(lines, 'eng_Latn', 'ind_Latn')
for s, r in zip(lines, res):
    p = postprocess_nllb_indonesian(r, src_lang='eng_Latn')
    sys.stdout.buffer.write(f"EN: {s}\n  -> NLLB: {r}\n  -> Post: {p}\n".encode("utf-8"))
