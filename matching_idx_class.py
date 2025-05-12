import os
import json

# VOCdevkit 경로 설정
voc_path = './VOCdata/VOCdevkit'
dataset_path = os.path.join(voc_path, 'VOC2007')
metadata_path = os.path.join(dataset_path, 'ImageSets', 'Main')

class_files = sorted(os.listdir(metadata_path))
class_names = []
for class_file in class_files:
    if '_' in class_file:
        class_name = class_file.split('_')[0]
        if class_name not in class_names:
            class_names.append(class_name)
            
class2idx_dict, idx2class_dict = {}, {}
for class_idx, class_name in enumerate(class_names):
    class2idx_dict[class_name] = class_idx
    idx2class_dict[class_idx] = class_name
    
with open(os.path.join(dataset_path, 'class2idx.json'), 'w') as f:
    json.dump(class2idx_dict, f, indent=4)
with open(os.path.join(dataset_path, 'idx2class.json'), 'w') as f:
    json.dump(idx2class_dict, f, indent=4)