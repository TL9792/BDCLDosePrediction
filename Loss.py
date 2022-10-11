import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.utils.data.dataset import ConcatDataset
from MultiBeamVoting import reconstruct
from OverlapConsistency import consistloss
from EdgeEnhancement.GradientLoss import gradient_loss
from DVH_Calibration.Value_basedDVHLoss import dvh_loss_v
from DVH_Calibration.Criteria_basedDVHLoss import dvh_loss_c


def mae_loss(pd, gt, dose_mask):
    "Mean absolute error (MAE) loss"
    diff_abs = torch.abs(pd - gt)
    return diff_abs.sum() / dose_mask.sum()
  
  
def loss(pd_dose,gt_dose, dose_mask, roi, spacing, coarse_dose, gtimg, whole_mask):
    "training loss function"
    reimg = reconstruct(pd_dose, coarse_dose)
    doseloss = mae_loss(pd_dose, gt_dose, dose_mask)
    Lr = 0.5*mae_loss(reimg, gtimg, whole_mask)
    Lcs = 0.1*consistloss(pd_dose,dose_mask)
    cdvhloss = 0.1*dvh_loss_c(reimg,gtimg, whole_mask)
    fdvhloss = 0.5*dvh_loss_v(reimg, gtimg, roi, spacing)
    Le = 0.5*gradient_loss(pd_dose, gt_dose, dose_mask)
    return (doseloss+Lr+Lcs+cdvhloss+fdvhloss+Le),reimg


