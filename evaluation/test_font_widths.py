import sys
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (100, 100))
draw = ImageDraw.Draw(img)

for sz in range(12, 6, -1):
    f = ImageFont.truetype("C:/Windows/Fonts/comicbd.ttf", sz)
    bb1 = draw.textbbox((0, 0), "menyelesaikan", font=f)
    bb2 = draw.textbbox((0, 0), "mengantuk", font=f)
    print(f"Size {sz}: 'menyelesaikan' width={bb1[2]-bb1[0]}, 'mengantuk' width={bb2[2]-bb2[0]}")
