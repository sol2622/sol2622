import base64

# Base64 데이터
base64_str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."

# Base64 → PNG 변환
image_data = base64.b64decode(base64_str.split(",")[1])

# 이미지 저장
with open("output/result.png", "wb") as f:
    f.write(image_data)

print("✅ 변환된 PNG 저장 완료: output/result.png")
