from flask import Flask, request, jsonify
import os
import sys

# 현재 스크립트의 디렉터리 기준으로 `models` 폴더 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
from models.generator import Generator  # 배경 제거 모델 로드

# Flask 앱 초기화
app = Flask(__name__)

# 🔹 저장 폴더 설정
UPLOAD_FOLDER = "uploads/"
OUTPUT_FOLDER = "outputs/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ 배경 제거 모델 로드 (서버 실행 시 한 번만 로드)
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location="cpu"))
generator.eval()

# ✅ 배경 제거 함수
def remove_background(input_png, output_png):
    """딥러닝 모델을 활용하여 배경 제거"""
    image = cv2.imread(input_png, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, "이미지를 불러올 수 없습니다."

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])  # 원본 크기 저장
    image_resized = cv2.resize(image, (900, 900))  # 모델 입력 크기로 조정

    # ✅ 이미지 정규화 및 변환
    image_tensor = torch.tensor(image_resized, dtype=torch.float32) / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    # ✅ 배경 제거 수행
    with torch.no_grad():
        mask = generator(image_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    # ✅ 마스크 후처리
    mask = (mask + 1) / 2  # [-1,1] → [0,1] 변환
    mask = (mask * 255).astype(np.uint8)  # 0~255 변환

    # ✅ 마스크가 3채널이면 단일 채널로 변환
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # ✅ 원본 크기에 맞게 리사이즈
    mask = cv2.resize(mask, original_size)

    # 🔥 **객체 내부 보호 (흰색 부분이 배경으로 오인되지 않도록)**
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 내부 폐색 영역 감지
    mask_filled = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)  # 객체 내부 유지

    # ✅ **객체 내부의 작은 흰색 영역 감지 및 보호**
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # 작은 객체 내부 영역 보호
            cv2.drawContours(mask_filled, [cnt], -1, 255, thickness=cv2.FILLED)

    # ✅ GrabCut을 위한 마스크 변환 (전경 픽셀 확장)
    _, binary_mask = cv2.threshold(mask_filled, 77, 255, cv2.THRESH_BINARY)
    binary_mask = np.where(binary_mask > 128, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)

    # 🔥 **전경 픽셀 부족 시 자동 확장**
    if np.sum(binary_mask == cv2.GC_FGD) < 3000:
        binary_mask = cv2.dilate(binary_mask, np.ones((28, 28), np.uint8), iterations=2)
        binary_mask = cv2.erode(binary_mask, np.ones((7, 7), np.uint8), iterations=1)

    # ✅ GrabCut을 이용해 배경을 점진적으로 제거
    mask_grabcut = np.full_like(mask_filled, cv2.GC_PR_BGD, dtype=np.uint8)
    mask_grabcut[binary_mask == cv2.GC_FGD] = cv2.GC_FGD

    # 🔥 **mask_grabcut을 np.uint8로 변환 (오류 방지)**
    mask_grabcut = mask_grabcut.astype(np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 🚀 **GrabCut 적용**
    cv2.grabCut(image, mask_grabcut, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    binary_mask = np.where((mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 🔥 **객체 끝부분을 부드럽게 블러 처리**
    binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 1)

    # ✅ 배경을 투명하게 변환 (알파 채널 추가)
    image_with_alpha = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)

    # 🔥 크기 불일치 방지 (알파 채널 적용 전 크기 확인)
    if binary_mask.shape != (image_with_alpha.shape[0], image_with_alpha.shape[1]):
        binary_mask = cv2.resize(binary_mask, (image_with_alpha.shape[1], image_with_alpha.shape[0]))

    image_with_alpha[:, :, 3] = binary_mask  # Alpha 채널에 블러 처리된 마스크 적용

    # ✅ 저장
    cv2.imwrite(output_png, image_with_alpha)
    return output_png, "✅ 배경 제거 완료"

# ✅ 파일 업로드 API 엔드포인트
@app.route("/remove_bg", methods=["POST"])
def remove_background_api():
    """파일을 업로드 받아 배경 제거 후 결과 파일 반환"""
    if "image" not in request.files:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    file = request.files["image"]
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")

    file.save(input_path)  # 파일 저장

    # 배경 제거 실행
    result, msg = remove_background(input_path, output_path)
    if result is None:
        return jsonify({"error": msg}), 500

    return jsonify({"output_file": output_path})

# ✅ Flask 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
