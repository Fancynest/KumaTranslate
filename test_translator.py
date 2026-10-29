import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.translator import MangaTranslator

def test():
    # Inisialisasi translator (default target: Bahasa Indonesia)
    translator = MangaTranslator()

    japanese_texts = [
        "お前はもう死んでいる",
        "こんにちは！元気ですか？",
        "助けて！誰か！",
        "俺は海賊王になる男だ！"
    ]

    print("\n" + "=" * 50)
    print("PENGUJIAN TRANSLATOR JEPANG -> INDONESIA & INGGRIS")
    print("=" * 50)

    for text in japanese_texts:
        indo = translator.translate(text, tgt_lang="ind_Latn")
        eng = translator.translate(text, tgt_lang="eng_Latn")
        print(f"Jepang   : {text}")
        print(f"Indonesia: {indo}")
        print(f"Inggris  : {eng}")
        print("-" * 50)

if __name__ == "__main__":
    test()
