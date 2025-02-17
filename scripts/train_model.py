import os
import sys
# 경로 설정 (모델 폴더 인식)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from scripts.dataset.data_preprocessing import BackgroundRemovalDataset

# ✅ 하이퍼파라미터 설정
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0002

# ✅ 데이터셋 로드
dataset = BackgroundRemovalDataset(image_dir="dataset/images", mask_dir="dataset/masks")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 초기화
generator = Generator()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

# ✅ 모델 학습
for epoch in range(EPOCHS):
    total_loss = 0
    for images, masks in train_loader:
        optimizer.zero_grad()

        outputs = generator(images)  # 모델 예측 결과
        outputs = torch.mean(outputs, dim=1, keepdim=True)  # 🔥 3채널 → 1채널 변환

        # 🔥 Sigmoid 활성화 함수 추가 (출력값을 0~1 범위로 조정)
        outputs = torch.sigmoid(outputs)

        # 🔥 마스크 크기 변환 (출력 크기와 맞추기)
        masks = torch.nn.functional.interpolate(masks, size=(512, 512), mode='bilinear', align_corners=False)
        masks = masks / 255.0 if masks.max() > 1 else masks  # 🔥 마스크 0~1 정규화

        loss = criterion(outputs, masks)  # 정답 마스크와 비교하여 손실 계산
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# ✅ 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "models/generator.pth")
print("✅ 모델 학습 완료 및 저장됨: models/generator.pth")
