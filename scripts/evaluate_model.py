import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.generator import Generator
from scripts.dataset.data_preprocessing import BackgroundRemovalDataset

# 모델 로드
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location=torch.device("cpu")))
generator.eval()

# 데이터 로드
dataset = BackgroundRemovalDataset(image_dir="dataset/images", mask_dir="dataset/masks")
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 평가 수행
mse_loss = nn.MSELoss()
total_loss = 0

for images, masks in data_loader:
    images = images.permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        outputs = generator(images)
        outputs = (outputs + 1) / 2  # 🔥 [-1,1] → [0,1] 변환

    loss = mse_loss(outputs, masks.unsqueeze(1).float() / 255.0)
    total_loss += loss.item()

average_loss = total_loss / len(data_loader)
print(f"🔥 평가 완료: 평균 손실(MSE) = {average_loss:.4f}")
