import sys
import os
# ✅ 현재 스크립트의 디렉터리를 기준으로 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import torch
import argparse
import numpy as np
from models.generator import Generator  # ✅ 모델 불러오기

# ✅ 인자값 설정
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
parser.add_argument("--output", type=str, required=True, help="출력 이미지 경로")
args = parser.parse_args()

# ✅ 모델 로드
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location=torch.device("cpu")))
generator.eval()

# ✅ 이미지 로드 및 전처리
image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)  # 원본 이미지 (RGBA 지원 가능)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
image_resized = cv2.resize(image_rgb, (512, 512))  # 모델 입력 크기로 변환

# ✅ 모델 입력 형식으로 변환
image_tensor = torch.tensor(image_resized, dtype=torch.float32) / 255.0
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] → [1, C, H, W]

# ✅ 배경 제거 수행 (마스크 예측)
with torch.no_grad():
    mask = generator(image_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

# 🔥 **마스크 값 조정 (중요)**
mask = (mask - mask.min()) / (mask.max() - mask.min())  # 0~1 스케일 변환
mask = (mask * 255).astype("uint8")  # 0~255로 변환

# ✅ 원본 크기로 다시 변경 (해상도 유지)
mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

# ✅ 투명한 배경 생성
_, binary_mask = cv2.threshold(mask_resized, 128, 255, cv2.THRESH_BINARY)  # 128 기준으로 이진화
binary_mask = binary_mask / 255.0  # 0~1 스케일 변환

# ✅ 원본 이미지에서 배경 제거 (객체만 유지)
image_alpha = image_rgb * binary_mask[:, :, None]  # 마스크를 적용하여 객체만 남김
image_alpha = image_alpha.astype("uint8")

# ✅ RGBA 채널 추가 (투명 배경)
result = np.dstack((image_alpha, (binary_mask * 255).astype("uint8")))

# ✅ 투명 PNG로 저장
output_path = args.output.replace(".jpg", ".png")  # 확장자 자동 변경
cv2.imwrite(output_path, result)
print(f"✅ 배경 제거 완료! 결과 저장: {output_path}")
