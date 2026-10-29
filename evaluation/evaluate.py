import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any
from PIL import Image

# Setup encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import BubbleDetector

def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box [x, y, w, h].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    return interArea / unionArea if unionArea > 0 else 0.0

def evaluate_detector(
    images_dir: str = "evaluation/images",
    gt_path: str = "evaluation/ground_truth.json",
    iou_threshold: float = 0.50,
    save_debug: bool = False,
    target_type: str = "text"  # 'text', 'bubble', atau 'both'
) -> Dict[str, Any]:
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"File Ground Truth tidak ditemukan: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    detector = BubbleDetector()
    modes_to_run = ["text_bbox", "bubble_bbox"] if target_type == "both" else ([f"{target_type}_bbox"] if not target_type.endswith("_bbox") else [target_type])

    final_results = {}

    for mode in modes_to_run:
        mode_label = "Klaster Teks (text_bbox)" if mode == "text_bbox" else "Batas Balon (bubble_bbox)"
        print("=" * 86)
        print(f"{'EVALUASI DETEKTOR - ' + mode_label:^86}")
        print(f"{'Dataset: 26 Halaman Nyata | Ambang Batas IoU: ' + str(iou_threshold):^86}")
        print("=" * 86)
        print(f"{'No':<3} | {'Nama File':<24} | {'GT':<4} | {'Pred':<5} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec (%)':<9} | {'Rec (%)':<8} | {'F1 (%)':<7}")
        print("-" * 86)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_matched_ious = []
        page_results = []
        t0 = time.time()

        for idx, (img_name, gt_boxes) in enumerate(sorted(ground_truth.items())):
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                continue

            predictions = detector.detect(img_path, debug=save_debug, debug_dir="evaluation/debug_overlays")
            pred_boxes = [p[mode] for p in predictions]

            # Greedy bipartite matching berbasis IoU
            matched_gt = set()
            matched_pred = set()
            page_ious = []

            # Hitung matrix IoU
            iou_matrix = []
            for p_idx, p_box in enumerate(pred_boxes):
                for g_idx, g_box in enumerate(gt_boxes):
                    iou = calculate_iou(p_box, g_box)
                    if iou >= iou_threshold:
                        iou_matrix.append((iou, p_idx, g_idx))

            iou_matrix.sort(key=lambda x: x[0], reverse=True)

            for iou, p_idx, g_idx in iou_matrix:
                if p_idx not in matched_pred and g_idx not in matched_gt:
                    matched_pred.add(p_idx)
                    matched_gt.add(g_idx)
                    page_ious.append(iou)

            tp = len(matched_pred)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_matched_ious.extend(page_ious)

            prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            print(f"{idx+1:<3} | {img_name:<24} | {len(gt_boxes):<4} | {len(pred_boxes):<5} | {tp:<4} | {fp:<4} | {fn:<4} | {prec:9.1f} | {rec:8.1f} | {f1:7.1f}")
            page_results.append({
                "image": img_name,
                "gt": len(gt_boxes),
                "pred": len(pred_boxes),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1
            })

        duration = time.time() - t0
        overall_prec = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
        mean_iou = (sum(all_matched_ious) / len(all_matched_ious)) if all_matched_ious else 0.0

        print("=" * 86)
        print(f"RANGKUMAN EVALUASI ({mode_label}) ({len(page_results)} Halaman Selesai dalam {duration:.1f}s):")
        print(f"  * Total Ground Truth Bubbles : {total_tp + total_fn}")
        print(f"  * Total Prediksi Model       : {total_tp + total_fp}")
        print(f"  * True Positives (TP)        : {total_tp}")
        print(f"  * False Positives (FP)       : {total_fp}")
        print(f"  * False Negatives (FN)       : {total_fn}")
        print(f"  * Overall Precision          : {overall_prec:.2f}%")
        print(f"  * Overall Recall             : {overall_rec:.2f}%")
        print(f"  * Overall F1-Score           : {overall_f1:.2f}%")
        print(f"  * Mean IoU (True Positives)  : {mean_iou:.3f}")
        print("=" * 86 + "\n")

        final_results[mode] = {
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1,
            "mean_iou": mean_iou
        }

    return final_results

if __name__ == "__main__":
    target = "both" if "--both" in sys.argv else ("bubble" if "--bubble" in sys.argv else "both")
    debug_flag = "--debug" in sys.argv
    evaluate_detector(target_type=target, save_debug=debug_flag)
