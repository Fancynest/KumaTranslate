import sys
sys.path.insert(0, ".")
from src.text_utils import rejoin_hyphenated_words, split_glued_words, correct_spelling_latin, normalize_latin_case, post_process_latin_ocr

# What was the raw OCR text of bubble 8?
# Was it "They done need to know that!" or "THEY DONE NEED TO KNOW THAT!"?
for text in ["THEY DONE NEED TO KNOW THAT!", "They done need to know that!", "THEY DON'T NEEDTO KNOW THAT!"]:
    print(f"INPUT: '{text}'")
    out, q = post_process_latin_ocr(text)
    print(f"  OUTPUT: '{out}'")
