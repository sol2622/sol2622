import torch
import torch.nn as nn

class Generator(nn.Module):
    """ GAN Generator """
 
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # (C_in=3, C_out=64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # (C_in=64, C_out=3)
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)  # 이미지는 (B, C, H, W) 형식으로 전달됨
