import torch
import torchvision.transforms.v2 as T2

from utils.custom_data import PascalVoc2007Yolov1Dataset
from utils.loss import divide_tensor

transform = T2.Compose([
    T2.ToImage()
])
C, B, S = 20, 2, 7

train_ds = PascalVoc2007Yolov1Dataset('.', 'train', transform=transform)
img, gt = train_ds[0]
output = torch.randn(size=(S, S, C + 5*B))

label_prob, label_x, label_y, label_w, label_h, label_conf = divide_tensor(gt, C, B)
pred_prob, pred_x, pred_y, pred_w, pred_h, pred_conf = divide_tensor(output, C, B)

# (7, 7, 2) -> (7, 7) -> (7, 7, 1)
obj_exists_cell_mask = label_conf[..., 0].unsqueeze(-1)
pred_prob_masked = pred_prob * obj_exists_cell_mask
print(pred_prob_masked.shape)

loss_clf = torch.sum((label_prob - pred_prob_masked)**2)

preds_x_mask = pred_x + obj_exists_cell_mask
preds_y_mask = pred_y + obj_exists_cell_mask
preds_w_mask = pred_w + obj_exists_cell_mask
preds_h_mask = pred_h + obj_exists_cell_mask











