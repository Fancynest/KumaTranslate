import sys
sys.path.insert(0, ".")
from PIL import Image
from src.detector import BubbleDetector

img = Image.open('temp/80f46fe3/extracted/page_001.png') if False else None
# Let's find where page 2 of Futaiwa is located in temp/
