import torch

def apply_label_smoothing(label_tensor, smooth_bg=0.05, smooth_lane=0.95):
    return torch.where(
        label_tensor == 1,
        torch.full_like(label_tensor, smooth_lane),
        torch.full_like(label_tensor, smooth_bg)
    )