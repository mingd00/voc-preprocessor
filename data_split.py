import os
import config
import numpy as np

# 디렉토리 경로 설정
dataset_path = config.dataset_path
dataset_path = os.path.join(dataset_path,'VOCdevkit', 'VOC2007')
annot_path = os.path.join(dataset_path,'Annotations')

# 저장 경로 생성
split_path = os.path.join(dataset_path, "TrainValTestIDs")
os.makedirs(split_path, exist_ok=True)

# XML 파일 목록 가져오기
all_ids = [f[:-4] for f in os.listdir(annot_path) if f.endswith(".xml")]

# 개수 계산
total = len(all_ids)
train_end = int(total * config.train_ratio)
val_end = train_end + int(total * config.val_ratio)

rng = np.random.default_rng(0) # local한 Random Seed
rng.shuffle(all_ids)

train_ids = all_ids[:train_end]
val_ids = all_ids[train_end:val_end]
test_ids = all_ids[val_end:]

# 각 ID 목록을 파일로 저장
def save_ids(file_path, ids):
    with open(file_path, 'w') as f:
        for id_ in ids:
            f.write(f"{id_}\n")

save_ids(os.path.join(split_path, "train_ids.txt"), train_ids)
save_ids(os.path.join(split_path, "val_ids.txt"), val_ids)
save_ids(os.path.join(split_path, "test_ids.txt"), test_ids)

print(f"Saved {len(train_ids)} train IDs, {len(val_ids)} val IDs, {len(test_ids)} test IDs.")
