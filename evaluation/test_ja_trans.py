import sys
sys.path.insert(0, ".")
from src.translator import MangaTranslator

t = MangaTranslator()
lines = [
    '手土産を持ってきてやったぞ',
    'うちはサスケの娘にして',
    'あのボルトにとっての最大の弱みだ',
    'この娘の事となると様子が変わるんだ',
    '冷静さを欠き判断力を鈍らせる',
    '良い策を教えてやる'
]
res = t.translate_batch(lines, 'jpn_Jpan', 'ind_Latn')
for s, r in zip(lines, res):
    sys.stdout.buffer.write(f"{s} -> {r}\n".encode("utf-8"))
