import torch

def divide_tensor(input_tensor, C, B):
    prob, bbox_conf = input_tensor[..., :C], input_tensor[..., C:]

    x_indices = torch.arange(B) * 5
    y_indices = x_indices + 1
    w_indices = x_indices + 2
    h_indices = x_indices + 3
    conf_indices = x_indices + 4

    x, y = bbox_conf[..., x_indices], bbox_conf[..., y_indices]
    w, h = bbox_conf[..., w_indices], bbox_conf[..., h_indices]
    conf = bbox_conf[..., conf_indices]

    return prob, x, y, w, h, conf