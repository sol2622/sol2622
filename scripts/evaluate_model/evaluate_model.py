import sys
import os
# 현재 스크립트의 디렉터리 경로를 추가하여 models 폴더를 인식하도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader

from models.generator import Generator  # 🔥 여기서 이제 models를 찾을 수 있음
from scripts.dataset.data_preprocessing import BackgroundRemovalDataset

# ✅ 평가 결과 저장 폴더 생성
output_folder = "output/evaluation_results"
os.makedirs(output_folder, exist_ok=True)

# ✅ GPU 사용 가능하면 CUDA 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()  # 평가 모드로 설정

# ✅ 데이터 로드
dataset = BackgroundRemovalDataset(root_dir="dataset/images")
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ✅ 평가 실행
mse_loss = nn.MSELoss()
total_loss = 0

for idx, images in enumerate(data_loader):
    images = images.permute(0, 3, 1, 2).float() / 255.0  # (B, H, W, C) → (B, C, H, W)
    images = images.to(device)

    with torch.no_grad():
        outputs = generator(images)
        outputs = (outputs + 1) / 2  # Generator의 출력을 0~1로 변환

    loss = mse_loss(outputs, images)  # MSE 손실 계산
    total_loss += loss.item()

    # ✅ 결과 이미지 저장
    output_image = outputs[0].cpu().numpy().transpose(1, 2, 0) * 255
    output_image = output_image.astype(np.uint8)
    output_path = os.path.join(output_folder, f"result_{idx}.png")
    cv2.imwrite(output_path, output_image)

    # ✅ 결과 이미지 확인 (디버깅용)
    print(f"✔️ 저장 완료: {output_path}")

# ✅ 평균 손실 출력
average_loss = total_loss / len(data_loader)
print(f"🔥 평가 완료: 평균 손실(MSE) = {average_loss:.4f}")
