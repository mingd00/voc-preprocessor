import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PascalVoc2007Dataset(Dataset):
    def __init__(self, dataset_path, dataset_type='train', transform=None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.transform = transform 
        
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
        
        if self.transform:
            img, annot = self.transform(img, annot)
            
        return img, annot
        
    def __len__(self):
        return len(self.ids)
    
class PascalVoc2007Yolov1Dataset(PascalVoc2007Dataset):
    def __init__(self, dataset_path, dataset_type='train', transform=None):
        super().__init__(dataset_path, dataset_type, transform)
        self.S = 7
        self.C = 20
        self.B = 2
    
    def _transform_bboxes(self, img, annot):
        bboxes = annot['boxes']
        bboxes_ = bboxes.clone()
        bboxes_[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bboxes_[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        bboxes_[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes_[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        bboxes = bboxes_

        # calculate relative coordinates
        S = 7
        _, H, W = img.shape
        cell_H, cell_W = H / S, W / S
        x_center_rel = bboxes_[:, 0] / cell_W
        x_center_loc = bboxes_[:, 0] // cell_W

        bboxes[:, 0] = x_center_rel - x_center_loc
        y_center_rel = bboxes[:, 1] / cell_H
        y_center_loc = bboxes_[:, 1] // cell_H

        bboxes[:, 1] = y_center_rel - y_center_loc
        bboxes[:, 2] /= W
        bboxes[:, 3] /= H
        
        annot['boxes'] = bboxes
        return annot, x_center_loc, y_center_loc
    
    def __getitem__(self, sample_idx):
        img, annot = super().__getitem__(sample_idx)
        annot, x_center_loc, y_center_loc = self._transform_bboxes(img, annot)
        labels, bboxes = annot['labels'], annot['boxes']
        
        depth = self.C + self.B * 5
        gt = torch.zeros(self.S, self.S, depth)
        gt[y_center_loc.long(), x_center_loc.long(), labels.long()] = 1 # 그 클래스에 해당하는 부분만 1로 바꿔라, x랑 y 순서 주의 
        
        n_bboxes = len(bboxes)
        bboxes = torch.hstack([bboxes, torch.ones(size=(n_bboxes, 1))])
        bboxes = torch.hstack([bboxes for _ in range(self.B)])
        
        gt[y_center_loc.long(), x_center_loc.long(), self.C:] = bboxes
        
        return img, gt


if __name__ == "__main__":
    train_ds = PascalVoc2007Dataset('.', 'train')
    img, annot = train_ds[0]
    # print(img.size)
    # print(annot)
    # val_ds = PascalVoc2007Dataset(dataset_path, 'val')
    # test_ds = PascalVoc2007Dataset(dataset_path, 'test')

    
    