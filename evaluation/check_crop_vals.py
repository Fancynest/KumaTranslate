import cv2
import numpy as np
from PIL import Image

img = Image.open("temp/59c91ad5/extracted/page_001.png")
gray = np.array(img.convert("L"))
# bubble 0: bbox=(450, 0, 79, 114)
crop = gray[0:114, 450:450+79]
print("crop shape:", crop.shape)
print("crop[41, 50:65]:", crop[41, 50:65])
print("crop[42, 50:65]:", crop[42, 50:65])
print("crop[45, 50:65]:", crop[45, 50:65])
print("crop[50, 50:65]:", crop[50, 50:65])
print("mean value of crop[41:70, 50:75]:", np.mean(crop[41:70, 50:75]))
