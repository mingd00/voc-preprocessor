from utils.custom_data import PascalVoc2007Dataset
import matplotlib.pyplot as plt
from utils.vls import draw_bbox
import torchvision.transforms.v2 as T2

transform = T2.Compose([
    T2.ToImage(),
    T2.Resize(size=(448, 448)),
    T2.RandomHorizontalFlip(1),
    T2.RandomAffine(translate=(0.2, 0.2), scale=(0.8, 1.2), degrees=0)
])

train_ds = PascalVoc2007Dataset('.', 'train', transform=transform)
img, annot = train_ds[2]
img_t, annot_t = transform(img, annot)

fig, axes = plt.subplots(1, 2, dpi=200, layout='constrained')

draw_bbox(axes[0], img.permute(1, 2, 0), annot)
draw_bbox(axes[1], img_t.permute(1, 2, 0), annot_t)
plt.show()

# matplotlib에서 imread -> (h, w, 3) ndarray 
# torch (3, h, w)