import sys
sys.path.insert(0, ".")
from src.text_utils import post_process_latin_ocr

lines = [
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

for l in lines:
    cleaned = post_process_latin_ocr(l)
    print(f"RAW:  {l}")
    print(f"POST: {cleaned}")
    print()
