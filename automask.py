import cv2
import numpy as np

# 📌 원본 이미지 & 배경 제거된 이미지 로드
image = cv2.imread("dataset/images/image_01.png")
bg_removed = cv2.imread("dataset/masks/image_01.png", cv2.IMREAD_GRAYSCALE)  # 현재 잘못된 마스크

# 📌 올바른 마스크 생성 (흰색: 255, 검은색: 0)
_, binary_mask = cv2.threshold(bg_removed, 1, 255, cv2.THRESH_BINARY)

# 📌 저장
cv2.imwrite("dataset/masks/image_01_mask.png", binary_mask)
print("✅ 올바른 마스크 저장 완료!")