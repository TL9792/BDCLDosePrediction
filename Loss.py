import torch

def mae_loss(pd, gt, dose_mask):
  "Mean absolute error (MAE) loss"
    diff_abs = torch.abs(pd - gt)
    return diff_abs.sum() / dose_mask.sum()
