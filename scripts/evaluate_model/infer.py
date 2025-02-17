import sys
import os

# 현재 스크립트의 디렉터리 경로를 추가하여 models 폴더를 인식하도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import argparse
from models.generator import Generator  # 🚀 models 폴더에서 Generator 불러오기


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
parser.add_argument("--output", type=str, required=True, help="출력 이미지 경로")
args = parser.parse_args()

# ✅ 모델 로드
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth"))
generator.eval()

# ✅ 이미지 로드 및 전처리
image = cv2.imread(args.input)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
image = torch.tensor(image, dtype=torch.float32) / 255.0
image = image.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]

# ✅ 배경 제거 수행
output = generator(image).squeeze(0).permute(1, 2, 0).detach().numpy()
output = (output * 255).astype("uint8")

# ✅ 저장
cv2.imwrite(args.output, output)
print(f"✅ 배경 제거 완료: {args.output}")
