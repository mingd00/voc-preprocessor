import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PascalVoc2007Dataset(Dataset):
    def __init__(self, dataset_path, dataset_type='train'):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        
        self._set_paths()
        self._load_ids()
        
    def _set_paths(self):
        self.dataset_path = os.path.join(self.dataset_path, 'VOCdata', 'VOCdevkit', 'VOC2007')
        self.img_path = os.path.join(self.dataset_path, 'JPEGImages')
        self.annot_path = os.path.join(self.dataset_path, 'AnnotationsBoundingBoxes')
        self.id_path = os.path.join(self.dataset_path, 'TrainValTestIDs')
        
    def _load_ids(self):
        if self.dataset_type == 'train':
            id_file = os.path.join(self.id_path, 'train_ids.txt')
        elif self.dataset_type == 'val':
            id_file = os.path.join(self.id_path, 'val.txt')
        elif self.dataset_type == 'test':
            id_file = os.path.join(self.id_path, 'test_ids.txt')
            
        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
    
    def __getitem__(self, sample_idx): 
        id = self.ids[sample_idx]
        img_file = os.path.join(self.img_path, f"{id}.jpg")
        annot_file = os.path.join(self.annot_path, f"{id}.pt")

        img = Image.open(img_file)
        annot = torch.load(annot_file, weights_only=False)
        return img, annot
        
    def __len__(self):
        return len(self.ids)
    
if __name__ == "__main__":
    train_ds = PascalVoc2007Dataset('.', 'train')
    img, annot = train_ds[0]
    print(img.size)
    print(annot)
    # val_ds = PascalVoc2007Dataset(dataset_path, 'val')
    # test_ds = PascalVoc2007Dataset(dataset_path, 'test')
    
    