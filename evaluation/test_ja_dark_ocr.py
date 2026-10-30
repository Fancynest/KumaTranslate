import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.ocr import MultiLingualOCR

ocr = MultiLingualOCR()

# Create a synthetic black bubble with white Japanese text
img = np.zeros((160, 200, 3), dtype=np.uint8)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

# Try drawing Japanese text
font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 28)
draw.text((20, 20), "貴様を殺す", font=font, fill=(255, 255, 255))

dark_ja_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imwrite("evaluation/synthetic_dark_ja.png", dark_ja_cv)

# Test MangaOCR on normal dark image
res_normal = ocr.manga_ocr(img_pil).strip()
# Test MangaOCR on inverted image
inv_pil = Image.fromarray(255 - np.array(img_pil))
res_inv = ocr.manga_ocr(inv_pil).strip()

print("Normal (White on Black) -> MangaOCR:", res_normal.encode('unicode_escape').decode())
print("Inverted (Black on White) -> MangaOCR:", res_inv.encode('unicode_escape').decode())
