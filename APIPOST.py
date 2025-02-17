import requests

# Flask API 엔드포인트 URL
API_URL = "http://localhost:5000/remove_bg"

# 업로드할 이미지 파일 지정
file_path = "basetest/inputon/J326707-PDH-ITEM-1.png"

# API 요청 (파일 업로드 방식)
with open(file_path, "rb") as file:
    files = {"image": file}
    response = requests.post(API_URL, files=files)

# 🛠 응답 처리
if response.status_code == 200:
    try:
        result = response.json()
        print("✅ 변환된 이미지 경로:", result["output_file"])
    except requests.exceptions.JSONDecodeError:
        print("❌ JSON 디코딩 오류: 응답이 JSON 형식이 아닙니다.")
        print("📌 서버 응답 내용:", response.text)  # 서버 응답 내용을 직접 출력
else:
    print(f"❌ 오류 발생: {response.status_code}")
    print("📌 서버 응답 내용:", response.text)  # 서버 응답 내용을 직접 출력