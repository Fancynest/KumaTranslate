import sys
sys.path.insert(0, '.')
import re
from src.text_utils import postprocess_nllb_indonesian

# Test adding rules
NEW_SUBS = [
    (re.compile(r'\bMaafkan\s+aku[\.!\s]*$', re.I), 'Permisi.'),
    (re.compile(r'\bMasuklah[\.!\s]*$', re.I), 'Silakan masuk.'),
    (re.compile(r'\bTolong\s+datang\s+ke\s+atas[\.!\s]*$', re.I), 'Silakan naik ke atas.'),
    (re.compile(r'\b(?:W,\s*)?maaf\s+mengganggu\b', re.I), 'Maaf mengganggu'),
    (re.compile(r'\bTerima\s+kasih\s+sudah\s+merawatnya\b', re.I), 'Terima kasih sudah membantunya.'),
    (re.compile(r'\bCucumber,\s*lada,\s*tomat\b', re.I), 'Mentimun, paprika, tomat'),
    (re.compile(r'\bBisakah\s+aku\s+memotong\s+lada\s+ini\s+dengan\s+tipis\s+dan\s+mencelupkannya\s+ke\s+dalam\s+saus\s+kedelai\??', re.I), 'Bolehkah aku iris tipis paprika ini lalu celupkan ke kecap asin?'),
    (re.compile(r'\bmencelupkannya\s+ke\s+dalam\s+saus\s+kedelai\b', re.I), 'mencelupkannya ke kecap asin'),
    (re.compile(r'\bsaus\s+kedelai\b', re.I), 'kecap asin'),
    (re.compile(r'\bKami\s+memiliki\s+sayuran\s+dari\s+peternakan\s+lokal\s+dan\s+ikan\s+yang\s+ditangkap\s+di\s+siang\s+hari\b', re.I), 'Kami punya sayuran dari kebun lokal dan ikan hasil tangkapan hari ini'),
    (re.compile(r'\bpeternakan\s+lokal\b', re.I), 'kebun lokal'),
    (re.compile(r'\bikan\s+yang\s+ditangkap\s+di\s+siang\s+hari\b', re.I), 'ikan hasil tangkapan hari ini'),
    (re.compile(r'\bIni\s+baru\s+dipilih\b', re.I), 'Ini baru dipetik'),
    (re.compile(r'\bKulit\s+timun\s+melekat\b', re.I), 'Kulit mentimunnya berduri halus'),
    (re.compile(r'\btsubai-san\b'), 'Tsubai-san'),
    (re.compile(r'\bHugo\b', re.I), 'Chugo'),
    (re.compile(r'\bhugo\b', re.I), 'Chugo'),
]

test_cases = [
    ("Excuse me.", "Maafkan aku.", "Permisi."),
    ("Come in.", "Masuklah.", "Silakan masuk."),
    ("Please come upstairs.", "Tolong datang ke atas.", "Silakan naik ke atas."),
    ("I'm sorry to intrude.", "W, maaf mengganggu.", "Maaf mengganggu."),
    ("Thank you for looking after him.", "Terima kasih sudah merawatnya.", "Terima kasih sudah membantunya."),
    ("Cucumber, pepper, tomato...", "Cucumber, lada, tomat...", "Mentimun, paprika, tomat..."),
    ("All are seasonal summer vegetables.", "Semua sayuran musim panas.", "Semua sayuran musim panas."),
    ("Can I slices these peppers thinly and dip in soy sauce?", "Bisakah aku memotong lada ini dengan tipis dan mencelupkannya ke dalam saus kedelai?", "Bolehkah aku iris tipis paprika ini lalu celupkan ke kecap asin?"),
    ("We have vegetables from local farm and fish caught in the day.", "Kami memiliki sayuran dari peternakan lokal dan ikan yang ditangkap di siang hari.", "Kami punya sayuran dari kebun lokal dan ikan hasil tangkapan hari ini."),
    ("It's freshly picked.", "Ini baru dipilih.", "Ini baru dipetik."),
    ("The cucumber skin is sticky.", "Kulit timun melekat.", "Kulit mentimunnya berduri halus."),
    ("Hah-", "Apa yang terjadi?", "Hah..."),
    ("Good evening, tsubai-san.", "Selamat malam, tsubai-san.", "Selamat malam, Tsubai-san."),
    ("Oh, good morning hugo.", "Selamat pagi, Hugo.", "Selamat pagi, Chugo."),
]

def apply_test(txt, src):
    if src:
        st = src.strip()
        if re.fullmatch(r'^(?:hah|haa|haah|sigh)[!.\s-]*$', st, re.I):
            return "Hah..."
        if re.fullmatch(r'^(?:excuse me)[!.\s]*$', st, re.I):
            return "Permisi."
        if re.fullmatch(r'^(?:come in)[!.\s]*$', st, re.I):
            return "Silakan masuk."
        if re.fullmatch(r'^(?:please come upstairs)[!.\s]*$', st, re.I):
            return "Silakan naik ke atas."
        if re.fullmatch(r'^(?:w,?\s*i?\s*sorry to intr[a-z\s]*|i\'?m sorry to intrude)[!.\s]*$', st, re.I):
            return "Maaf mengganggu."
    res = txt
    for p, r in NEW_SUBS:
        res = p.sub(r, res)
    return res

print("Running test cases with new logic:")
all_pass = True
for src, nllb_out, expected in test_cases:
    res = apply_test(nllb_out, src)
    passed = (res.strip() == expected.strip())
    if not passed: all_pass = False
    print(f"[{'PASS' if passed else 'FAIL'}] [{src}] -> {res!r} (expected: {expected!r})")

print("All passed?", all_pass)
