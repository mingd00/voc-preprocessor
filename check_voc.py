import os

# VOCdevkit 경로 설정
voc_path = './VOCdata/VOCdevkit/VOC2007'
img_dir = os.path.join(voc_path, 'JPEGImages')
ann_dir = os.path.join(voc_path, 'Annotations')

# 이미지와 어노테이션 파일 리스트
img_files = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')]
ann_files = [f[:-4] for f in os.listdir(ann_dir) if f.endswith('.xml')]

img_set = set(img_files)
ann_set = set(ann_files)

# 일치 여부 확인
common = img_set & ann_set

print(f"이미지 개수: {len(img_files)}")
print(f"어노테이션 개수: {len(ann_files)}")
print(f"매칭되는 파일 개수: {len(common)}")

print("\n--- ID 000001 ~ 001000 존재 여부 확인 ---")

expected_ids = [f"{i:06d}" for i in range(1, 1001)]
missing_imgs = [id_ for id_ in expected_ids if id_ not in img_set]
missing_anns = [id_ for id_ in expected_ids if id_ not in ann_set]
missing_ids = sorted(set(missing_imgs + missing_anns))

print(f"전체 기대 ID 수: {len(expected_ids)}")
print(f"정상적으로 존재하는 ID 수: {len(expected_ids) - len(missing_ids)}")

if missing_ids:
    print(f"\n누락된 ID 수: {len(missing_ids)}")
    if missing_imgs:
        print(f" - 이미지 누락: {len(missing_imgs)}개")
    if missing_anns:
        print(f" - 어노테이션 누락: {len(missing_anns)}개")
    print("\n누락된 ID 목록:")
    print("\n".join(missing_ids))
else:
    print("\n모든 ID(000001~001000)가 정상적으로 존재합니다.")
