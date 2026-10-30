import sys
sys.path.insert(0, ".")
from src.text_utils import post_process_latin_ocr, count_english_words, calculate_ocr_quality
t = 'T! ALOCAL FISHERMAN.'
t_clean, q = post_process_latin_ocr(t)
print('t_clean:', repr(t_clean), 'words:', count_english_words(t_clean), 'q:', q)
