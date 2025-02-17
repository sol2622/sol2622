import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2

# 경로 설정 (모델 폴더 인식)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 모델 불러오기
from models.matting_model import MattingModel
from models.generator import Generator
from models.discriminator import Discriminator
from scripts.dataset.data_preprocessing import BackgroundRemovalDataset  # 데이터셋 불러오기

# 하이퍼파라미터 설정
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.0002

# 데이터 로드
dataset = BackgroundRemovalDataset(root_dir="dataset/images")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 설정
matting_model = MattingModel()
generator = Generator()
discriminator = Discriminator()

# 손실 함수 및 최적화 설정
criterion = nn.BCELoss()  # BCE를 유지하는 경우
# criterion = nn.L1Loss()  # L1 손실 함수 대체 가능

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# 🔥 학습 루프
for epoch in range(EPOCHS):
    for images in train_loader:
        optimizer_G.zero_grad()

        # [수정] 이미지를 올바른 형태로 변환 (B, C, H, W)
        images = images.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        images = images.float() / 255.0  # Normalize (0~1)

        outputs = generator(images)
        
        # [중요 수정] Generator의 출력을 0~1로 변환
        outputs = (outputs + 1) / 2

        loss = criterion(outputs, images)

        loss.backward()
        optimizer_G.step()

    print(f"✅ 학습 완료: Epoch [{epoch+1}/{EPOCHS}]")

# 모델 저장
torch.save(generator.state_dict(), "models/generator.pth")
torch.save(discriminator.state_dict(), "models/discriminator.pth")
print("✅ 모델 저장 완료: models/generator.pth, models/discriminator.pth")
