import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

im = Image.open('evaluation/check_tr_p2.png').convert("L")
arr = np.array(im)
# check rows 0 to 40 in that crop
print("Min values in top 40 rows:", [np.min(arr[r]) for r in range(0, 40, 5)])
