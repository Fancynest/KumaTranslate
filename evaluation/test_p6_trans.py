import sys
sys.path.insert(0, '.')
from src.translator import MangaTranslator

tr = MangaTranslator()
t1 = "Don't say that!"
t2 = "Not everyday we have people came all the way from Tokyo to live here."
res = tr.translate_batch([t1, t2], src_lang='eng_Latn', tgt_lang='ind_Latn')
print('t1 ->', res[0])
print('t2 ->', res[1])
