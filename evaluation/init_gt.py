import os
import cv2
import json
from PIL import Image

EVAL_DIR = "evaluation/images"
OUTPUT_GT = "evaluation/ground_truth.json"

# Ground truth annotations: mapping image_filename -> list of [x, y, w, h] of real speech bubbles
ground_truth = {}

# 1. Wikipe-tan Manga Page 1 (250x333)
# 4 dialogue speech bubbles
ground_truth["wikipe_page1.jpg"] = [
    [23, 76, 25, 45],    # Bubble 1: 係船柱！ (left panel)
    [158, 75, 45, 30],   # Bubble 2: 港のキノコ... (top right panel)
    [112, 226, 25, 30],  # Bubble 3: あれ？ (bottom left panel)
    [185, 212, 48, 55]   # Bubble 4: 船を港に係留する... (bottom right panel)
]

# 2. Screenshot Manga (462x602)
# Real dialogue bubbles on this page
ground_truth["screenshot_manga.png"] = [
    [305, 10, 48, 88],
    [385, 18, 35, 48],
    [265, 172, 50, 100],
    [128, 180, 120, 105],
    [25, 265, 35, 60],
    [298, 375, 38, 58],
    [38, 452, 60, 118]
]

# 3. Sample Manga Page (800x1000)
ground_truth["sample_manga.png"] = [
    [520, 80, 220, 160],
    [100, 420, 250, 180],
    [480, 720, 260, 200]
]

# 4. Multi-language test cards
ground_truth["sample_japanese.png"] = [[20, 20, 260, 110]]
ground_truth["test_jp.png"] = [[15, 15, 370, 90]]
ground_truth["test_ko.png"] = [[15, 15, 370, 90]]
ground_truth["test_zh.png"] = [[15, 15, 370, 90]]
ground_truth["test_en.png"] = [[15, 15, 370, 90]]

# Save initial GT template
with open(OUTPUT_GT, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Saved initial ground truth with {len(ground_truth)} annotated images.")
