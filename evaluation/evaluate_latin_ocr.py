import os
import sys
import json
from typing import List, Tuple, Dict, Any, Sequence
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def levenshtein_distance(s1: Sequence, s2: Sequence) -> int:
    """
    Menghitung Levenshtein Edit Distance antara dua sequence (karakter atau kata).
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )

    return dp[m][n]


def compute_cer(ref: str, hyp: str) -> Tuple[int, int]:
    """Mengembalikan (edit_distance_chars, total_ref_chars)."""
    dist = levenshtein_distance(list(ref), list(hyp))
    return dist, len(ref)


def compute_wer(ref: str, hyp: str) -> Tuple[int, int]:
    """Mengembalikan (edit_distance_words, total_ref_words)."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist, len(ref_words)


def run_evaluation(
    engine_name: str = "rapid",
    crops_dir: str = "evaluation/ocr_latin_crops",
    gt_file: str = "evaluation/ocr_latin_ground_truth.json",
    preprocess_fn = None,
    postprocess_fn = None
) -> Dict[str, Any]:
    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth: Dict[str, str] = json.load(f)

    if engine_name == "rapid":
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        def ocr_func(img_np):
            res, _ = engine(img_np)
            return " ".join([x[1] for x in res]).strip() if res else ""
    elif engine_name == "rapid_en":
        from rapidocr_onnxruntime import RapidOCR
        rec_model = "models/rapidocr_en/en_PP-OCRv3_rec_infer.onnx"
        rec_keys = "models/rapidocr_en/en_dict.txt"
        engine = RapidOCR(rec_model_path=rec_model, rec_keys_path=rec_keys)
        def ocr_func(img_np):
            res, _ = engine(img_np)
            return " ".join([x[1] for x in res]).strip() if res else ""
    elif engine_name == "easy":
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        def ocr_func(img_np):
            res = reader.readtext(img_np, detail=0)
            return " ".join(res).strip() if res else ""
    else:
        raise ValueError(f"Engine tidak dikenal: {engine_name}")

    total_char_dist = 0
    total_ref_chars = 0
    total_word_dist = 0
    total_ref_words = 0

    per_sample_details = []

    for filename, ref_text in ground_truth.items():
        img_path = os.path.join(crops_dir, filename)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        if preprocess_fn:
            img = preprocess_fn(img)

        img_np = np.array(img)
        pred_text = ocr_func(img_np)
        if postprocess_fn:
            pred_text = postprocess_fn(pred_text)

        c_dist, c_len = compute_cer(ref_text, pred_text)
        w_dist, w_len = compute_wer(ref_text, pred_text)

        total_char_dist += c_dist
        total_ref_chars += c_len
        total_word_dist += w_dist
        total_ref_words += w_len

        sample_cer = (c_dist / c_len) if c_len > 0 else 0.0
        sample_wer = (w_dist / w_len) if w_len > 0 else 0.0

        per_sample_details.append({
            "file": filename,
            "ref": ref_text,
            "pred": pred_text,
            "cer": round(sample_cer * 100, 2),
            "wer": round(sample_wer * 100, 2)
        })

    overall_cer = (total_char_dist / total_ref_chars * 100) if total_ref_chars > 0 else 0.0
    overall_wer = (total_word_dist / total_ref_words * 100) if total_ref_words > 0 else 0.0

    return {
        "engine": engine_name,
        "total_samples": len(per_sample_details),
        "overall_cer": round(overall_cer, 2),
        "overall_wer": round(overall_wer, 2),
        "total_ref_chars": total_ref_chars,
        "total_char_errors": total_char_dist,
        "total_ref_words": total_ref_words,
        "total_word_errors": total_word_dist,
        "details": per_sample_details
    }


if __name__ == "__main__":
    engine_arg = "rapid"
    use_postprocess = False
    for arg in sys.argv[1:]:
        if arg == "--postprocess":
            use_postprocess = True
        elif not arg.startswith("--"):
            engine_arg = arg

    post_fn = None
    if use_postprocess:
        from src.text_utils import post_process_latin_ocr
        # Pass dummy chapter counts or empty dict
        def post_fn(t):
            cleaned, _ = post_process_latin_ocr(t)
            return cleaned

    results = run_evaluation(engine_name=engine_arg, postprocess_fn=post_fn)
    label = f"{engine_arg.upper()} {'+ POSTPROCESS' if use_postprocess else ''}"
    print(f"\n{'='*60}")
    print(f"HASIL EVALUASI OCR LATIN: {label}")
    print(f"{'='*60}")
    print(f"Total Balon Teruji : {results['total_samples']}")
    print(f"Character Error Rate (CER) : {results['overall_cer']}% ({results['total_char_errors']}/{results['total_ref_chars']} karakter)")
    print(f"Word Error Rate (WER)      : {results['overall_wer']}% ({results['total_word_errors']}/{results['total_ref_words']} kata)")
    print(f"{'='*60}\n")
