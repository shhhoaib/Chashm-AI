"""Test cv2 functionality."""
from PIL import Image
import numpy as np

img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
Image.fromarray(img).save("test_pil.png")
print("PIL image saved")

import cv2
try:
    read_back = cv2.imread("test_pil.png")
    print(f"cv2.imread works: shape={read_back.shape}")
    written = cv2.imwrite("test_cv2_out.png", read_back)
    print(f"cv2.imwrite works: {written}")
except Exception as e:
    print(f"cv2 error: {e}")
    print(f"cv2 type: {type(cv2)}")
    attrs = [x for x in dir(cv2) if not x.startswith("_")]
    print(f"First 20 attrs: {attrs[:20]}")
