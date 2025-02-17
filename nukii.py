import cv2
import numpy as np

# 📌 누끼 이미지 불러오기
image_path = "photoshop.png"  # 누끼 이미지 경로
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 📌 이진화 (Thresholding) 수행
_, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

# 📌 변환된 마스크 저장
mask_output_path = "dataset/masks/image_01_mask.png"  # 저장 경로
cv2.imwrite(mask_output_path, binary_mask)

print(f"✅ 마스크 변환 완료! 저장 경로: {mask_output_path}")
