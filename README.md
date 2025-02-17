<!-- python scripts/dataset/data_preprocessing.py
dataset/images/ 폴더에 있는 원본 이미지를 불러와서 학습에 필요한 형태로 변환
dataset/masks/ 폴더에 정답 마스크가 있어야 함

python scripts/train_model.py

Matting + GAN 모델을 학습하여 models/ 폴더에 모델 저장
학습된 모델:
models/matting_model.pth
models/generator.pth
models/discriminator.pth

python scripts/infer.py
input/ 폴더에 있는 이미지의 배경을 제거
output/ 폴더에 투명한 PNG로 저장 (output_transparent.png)

python scripts/evaluate_model.py
학습된 모델이 얼마나 정확한지 평가
IoU(Intersection over Union) 또는 MSE(Mean Squared Error) 같은 지표로 성능 측정 -->

# 1️⃣ 프로젝트 폴더 이동
cd C:\Users\USER\Desktop\AI-Background-Removal

# 2️⃣ 가상환경 활성화
venv\Scripts\activate

# 3️⃣ 패키지 설치
pip install -r requirements.txt

# 4️⃣ 데이터 전처리
python scripts/dataset/data_preprocessing.py

# 5️⃣ 모델 학습
python scripts/train_model.py

# 6️⃣ 모델 평가
python scripts/evaluate_model.py

# 7️⃣ 배경 제거 실행
API 실행 후 APIPOST로 PNG 파일 링크 넣어주고 실행
# sol2622
