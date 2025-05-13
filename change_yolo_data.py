import torch
import torchvision.transforms.v2 as T2

from utils.custom_data import PascalVoc2007Yolov1Dataset

transform = T2.Compose([
    T2.ToImage()
])

dataset = PascalVoc2007Yolov1Dataset('.', 'train', transform=transform)
print(dataset[2])