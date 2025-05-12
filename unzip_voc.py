import zipfile
import os

# 압축 파일 경로와 압축을 풀 위치 설정
zip_path = 'VOCdevkit.zip'        # 압축 파일 경로
extract_path = './VOCdata'     # 압축을 풀 디렉토리 경로

# 디렉토리가 없으면 생성
os.makedirs(extract_path, exist_ok=True)

# zip 파일 열고 압축 해제
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"{zip_path} 압축을 {extract_path} 폴더에 성공적으로 풀었습니다.")