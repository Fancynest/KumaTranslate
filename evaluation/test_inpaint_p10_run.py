import sys
sys.path.insert(0, '.')
import cv2
from PIL import Image
from src.detector import BubbleDetector
from src.inpainter import MangaInpainter

img_path = 'temp/ae483818/extracted/page_0010.png'
detector = BubbleDetector()
inpainter = MangaInpainter()

bubbles = detector.detect(img_path)
bboxes = [b['bbox'] for b in bubbles]
masks = [b.get('mask') for b in bubbles]
safe_bboxes = [b.get('safe_bbox') for b in bubbles]

img_pil = Image.open(img_path).convert('RGB')
cleaned = inpainter.clean_bubbles(img_pil, bboxes, masks=masks, safe_bboxes=safe_bboxes)
cleaned.save('evaluation/page10_cleaned.png')

# Crop around bubble 4 (862, 1205, 130, 93)
cleaned_cv = cv2.imread('evaluation/page10_cleaned.png')
crop_b4 = cleaned_cv[max(0, 1205-30):min(cleaned_cv.shape[0], 1205+93+30), max(0, 862-30):min(cleaned_cv.shape[1], 862+130+30)]
cv2.imwrite('evaluation/crop_b4_cleaned.png', crop_b4)
print("Saved crop_b4_cleaned.png")
