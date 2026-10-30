import sys
sys.path.insert(0, ".")
import re
from src.text_utils import post_process_latin_ocr
from src.translator import MangaTranslator
from src.text_utils import postprocess_nllb_indonesian

raw_lines = [
    "ACTUALLY OUR BUSINESS ISWMTHHIS PREDE. CESSOR.",
    "YEAH SOWHAT?",
    "YOUHAVEA MESSACE FORMY DADOR SOME- THING?",
    "WECAE HERE FOLLOWMING UZUMAKI NARUTO'S TRAILI",
    "50, FRANKLY WERE SURPRISED TOO.",
    "BUT THERE'S NOWAY SEVENTH WOULDBE HERE!",
    "HEY!",
    "Cuz THREE YEARS AGO, HC-",
    "THEY DON'T NEEDTO KNOW THAT!",
    "ISEE.",
    "WELL,HE DEFINITELY DOESNT SEEMTOBE ANYWHERE INTHIS VILLAGE.",
    "FORTHE OMLYNINE TAALS SIGNA- TURE..",
    "S COMING FROM THIS GIRL."
]

def pre_clean_comic_latin(text: str) -> str:
    t = text
    # Fix 50, -> So,
    t = re.sub(r'\b50,', 'So,', t)
    t = re.sub(r'\b50\b', 'So', t)
    # Fix common comic ligatures/ocr errors
    t = re.sub(r'\bISWMTHHIS\b|\bismthhis\b|\biswmthhis\b', 'is with his', t, flags=re.I)
    t = re.sub(r'\bPREDE\.\s*CESSOR\b', 'PREDECESSOR', t, flags=re.I)
    t = re.sub(r'\bMESSACE\b|\bmess\s+ace\b', 'message', t, flags=re.I)
    t = re.sub(r'\bFORMY\s+DADOR\b|\bform\s+dad\s+or\b', 'for my dad or', t, flags=re.I)
    t = re.sub(r'\bWECAE\s+HERE\b|\bweave\s+here\b', 'we came here', t, flags=re.I)
    t = re.sub(r'\bFOLLOWMING\b|\bfollow\s+ming\b', 'following', t, flags=re.I)
    t = re.sub(r'\bTRAILI\b|\btrail\s+I\b', 'trail.', t, flags=re.I)
    t = re.sub(r'\bNOWAY\b', 'no way', t, flags=re.I)
    t = re.sub(r'\bHC-\b', 'he-', t, flags=re.I)
    t = re.sub(r'\bOMLYNINE\b|\bomlynine\b', 'only nine', t, flags=re.I)
    t = re.sub(r'\bTAALS\b', 'tails', t, flags=re.I)
    t = re.sub(r'^[.·\s]*S\s+COMING\b', '...is coming', t, flags=re.I)
    return t

t_engine = MangaTranslator()

cleaned_lines = []
for r in raw_lines:
    pre = pre_clean_comic_latin(r)
    post_txt, q = post_process_latin_ocr(pre)
    cleaned_lines.append(post_txt)

print("CLEANED LINES FOR TRANSLATION:")
for c in cleaned_lines:
    print(" ", c)

translations = t_engine.translate_batch(cleaned_lines, 'eng_Latn', 'ind_Latn')
print("\nFINAL TRANSLATIONS:")
for c, tr in zip(cleaned_lines, translations):
    p = postprocess_nllb_indonesian(tr, src_lang='eng_Latn')
    sys.stdout.buffer.write(f"EN: {c}\n  -> ID: {p}\n\n".encode("utf-8"))
