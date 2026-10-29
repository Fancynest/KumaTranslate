import os
import sys
import unittest
import numpy as np
import cv2

# Pastikan path repo masuk ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import BubbleDetector

class TestSyntheticBubbles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = BubbleDetector(mode="auto")

    def _calculate_mask_iou(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        gt = gt_mask > 0
        pred = pred_mask > 0
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        return float(inter / union) if union > 0 else 0.0

    def _check_safe_bbox_containment(self, mask: np.ndarray, safe_bbox: tuple) -> float:
        sx, sy, sw, sh = safe_bbox
        self.assertGreater(sw, 0, "safe_bbox width harus > 0")
        self.assertGreater(sh, 0, "safe_bbox height harus > 0")
        sub_mask = mask[sy:sy+sh, sx:sx+sw]
        inside_pixels = np.sum(sub_mask > 0)
        return float(inside_pixels / (sw * sh))

    def test_oval_bubble(self):
        """
        Uji Balon Oval standar:
        Outline hitam elips lengkap di atas kanvas terang, dengan teks di tengah.
        """
        img = np.ones((400, 400), dtype=np.uint8) * 240
        cv2.ellipse(img, (200, 200), (90, 60), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (200, 200), (90, 60), 0, 0, 360, 0, 3)
        cv2.putText(img, "TEST DIALOG", (145, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

        gt_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(gt_mask, (200, 200), (87, 57), 0, 0, 360, 255, -1)

        text_bbox = (140, 190, 120, 25)
        roi_bbox = (105, 135, 190, 130)

        pred_mask, safe_bbox = self.detector._compute_bubble_mask(
            gray=img, text_bbox=text_bbox, roi_bbox=roi_bbox, font_size=16.0
        )

        iou = self._calculate_mask_iou(gt_mask, pred_mask)
        containment = self._check_safe_bbox_containment(pred_mask, safe_bbox)

        print(f"\n[Test 1: Balon Oval]")
        print(f"  Mask IoU: {iou:.4f}")
        print(f"  safe_bbox: {safe_bbox}")
        print(f"  safe_bbox containment in mask: {containment*100:.2f}%")

        self.assertGreaterEqual(iou, 0.85, f"IoU balon oval ({iou:.4f}) harus >= 0.85")
        self.assertEqual(containment, 1.0, f"safe_bbox harus 100% di dalam mask (dapat: {containment*100:.2f}%)")

    def test_open_tail_bubble(self):
        """
        Uji Balon dengan Ekor Terbuka / Celah pada Outline:
        Outline elips memiliki celah 6px pada ekor.
        Morphological opening harus memutus kebocoran ke kanvas luar.
        """
        img = np.ones((400, 400), dtype=np.uint8) * 235
        cv2.ellipse(img, (200, 180), (80, 55), 0, 0, 360, 255, -1)
        # Outline dengan celah 6px (0..354 derajat)
        cv2.ellipse(img, (200, 180), (80, 55), 0, 0, 354, 0, 3)
        cv2.putText(img, "OPEN TAIL", (150, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

        gt_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(gt_mask, (200, 180), (77, 52), 0, 0, 360, 255, -1)

        text_bbox = (145, 170, 110, 25)
        roi_bbox = (115, 120, 170, 130)

        pred_mask, safe_bbox = self.detector._compute_bubble_mask(
            gray=img, text_bbox=text_bbox, roi_bbox=roi_bbox, font_size=16.0
        )

        iou = self._calculate_mask_iou(gt_mask, pred_mask)
        containment = self._check_safe_bbox_containment(pred_mask, safe_bbox)

        print(f"\n[Test 2: Ekor Terbuka]")
        print(f"  Mask IoU: {iou:.4f}")
        print(f"  safe_bbox: {safe_bbox}")
        print(f"  safe_bbox containment in mask: {containment*100:.2f}%")

        self.assertGreaterEqual(iou, 0.85, f"IoU ekor terbuka ({iou:.4f}) harus >= 0.85")
        self.assertEqual(containment, 1.0, f"safe_bbox harus 100% di dalam mask (dapat: {containment*100:.2f}%)")

    def test_adjacent_bubbles(self):
        """
        Uji Dua Balon Berdekatan:
        Dua balon kata terpisah jarak sangat dekat (hanya 6px).
        Mask target hanya boleh mengambil balon target dan tidak menyatu ke balon tetangga.
        """
        img = np.ones((400, 400), dtype=np.uint8) * 240
        # Balon 1 (Target, kiri)
        cv2.ellipse(img, (130, 200), (60, 50), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (130, 200), (60, 50), 0, 0, 360, 0, 3)
        cv2.putText(img, "TARGET", (95, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        # Balon 2 (Tetangga, kanan, jarak 6px)
        cv2.ellipse(img, (256, 200), (60, 50), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (256, 200), (60, 50), 0, 0, 360, 0, 3)
        cv2.putText(img, "NEIGHBOR", (220, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        gt_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(gt_mask, (130, 200), (57, 47), 0, 0, 360, 255, -1)

        text_bbox = (90, 190, 80, 25)
        roi_bbox = (65, 145, 130, 110)

        pred_mask, safe_bbox = self.detector._compute_bubble_mask(
            gray=img, text_bbox=text_bbox, roi_bbox=roi_bbox, font_size=14.0
        )

        iou = self._calculate_mask_iou(gt_mask, pred_mask)
        containment = self._check_safe_bbox_containment(pred_mask, safe_bbox)

        # Pastikan tidak ada piksel balon tetangga yang bocor ke dalam mask
        neighbor_leak = np.sum(pred_mask[:, 220:] > 0)

        print(f"\n[Test 3: Dua Balon Berdekatan]")
        print(f"  Mask IoU: {iou:.4f}")
        print(f"  safe_bbox: {safe_bbox}")
        print(f"  safe_bbox containment in mask: {containment*100:.2f}%")
        print(f"  Neighbor leak pixels: {neighbor_leak}")

        self.assertGreaterEqual(iou, 0.88, f"IoU balon berdekatan ({iou:.4f}) harus >= 0.88")
        self.assertEqual(containment, 1.0, f"safe_bbox harus 100% di dalam mask (dapat: {containment*100:.2f}%)")
        self.assertEqual(neighbor_leak, 0, "Mask bocor ke balon tetangga!")

    def test_screentone_background(self):
        """
        Uji Balon dengan Latar Screentone:
        Di luar balon terdapat tekstur halftone/screentone komik (pola titik-titik).
        Deteksi harus mengekstrak interior putih balon tanpa terganggu pola tone di luar.
        """
        img = np.zeros((400, 400), dtype=np.uint8)
        # Buat pola screentone
        for y in range(0, 400, 4):
            for x in range(0, 400, 4):
                img[y, x] = 120
                img[y+1, x+1] = 160

        # Balon putih di tengah screentone
        cv2.ellipse(img, (200, 200), (85, 60), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (200, 200), (85, 60), 0, 0, 360, 0, 3)
        cv2.putText(img, "SCREENTONE", (135, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

        gt_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(gt_mask, (200, 200), (82, 57), 0, 0, 360, 255, -1)

        text_bbox = (130, 190, 140, 25)
        roi_bbox = (110, 135, 180, 130)

        pred_mask, safe_bbox = self.detector._compute_bubble_mask(
            gray=img, text_bbox=text_bbox, roi_bbox=roi_bbox, font_size=16.0
        )

        iou = self._calculate_mask_iou(gt_mask, pred_mask)
        containment = self._check_safe_bbox_containment(pred_mask, safe_bbox)

        print(f"\n[Test 4: Latar Screentone]")
        print(f"  Mask IoU: {iou:.4f}")
        print(f"  safe_bbox: {safe_bbox}")
        print(f"  safe_bbox containment in mask: {containment*100:.2f}%")

        self.assertGreaterEqual(iou, 0.88, f"IoU latar screentone ({iou:.4f}) harus >= 0.88")
        self.assertEqual(containment, 1.0, f"safe_bbox harus 100% di dalam mask (dapat: {containment*100:.2f}%)")

if __name__ == "__main__":
    unittest.main()
