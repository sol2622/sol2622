import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BackgroundRemovalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)  # 마스크는 흑백
        
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return image, mask


if __name__ == "__main__":
    print("✅ 데이터 전처리 완료")
