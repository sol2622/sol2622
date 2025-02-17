import cv2
import numpy as np

# 마스크 이미지 불러오기
mask = cv2.imread("dataset/masks/image_01.png", cv2.IMREAD_GRAYSCALE)

# 최소, 최대 값 출력 (0과 255인지 확인)
print(f"🔍 Mask Pixel Range: {mask.min()} ~ {mask.max()}")

# 마스크 시각화
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
