import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.utils.data.dataset import ConcatDataset

def dvh_loss_c(pd, gt, masks, spacing):
    """
    masks should have 10 ROI masks including
    oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
            'Esophagus', 'Larynx', 'Mandible'], targets=['PTV56', 'PTV63', 'PTV70']
    """

    # initial of the diff
    D_01_diff = 0
    mean_diff = 0
    D_99_diff = 0
    D_95_diff = 0
    D_1_diff  = 0
    n         = 0

    # get batch size and roi masks num
    batch_size = len(pd)
    roi_num = len(masks[0, ...])

    for batch in range(batch_size):

        # get voxel size
        voxelSize = torch.prod(spacing[batch, ...]).item()
        vixels_in_tenth_of_cc = max(1, round(100 / voxelSize))

        for roi in range(roi_num):
            
            # first flatten (128,128,128)
            pd_vec = pd[batch, 0, ...].flatten()
            gt_vec = gt[batch, 0, ...].flatten()
            mask_vec = masks[batch, roi, ...].flatten()

            # if mask has no value eg. all zero, then next roi
            if mask_vec.nonzero(as_tuple = False).numel() == 0:
                continue
                
            # oar masks need to calculate D_01_cc and mean as in the dose_evaluation.py                
            if roi < 7:

                # D_01_idx is the location of 99.5% intensity value
                length = mask_vec.nonzero(as_tuple = False).numel()
                fractional_volume_to_evaluate = 100 - vixels_in_tenth_of_cc / length * 100
                D_01_idx = fractional_volume_to_evaluate * 0.01

                # pred dose
                pd_region_vec = pd_vec * mask_vec
                pd_region_vec_nonzero = pd_region_vec[pd_region_vec.nonzero(as_tuple = False).detach()]
                if pd_region_vec_nonzero.numel() == 0:
                    continue
                D_01_pd = pd_region_vec_nonzero.topk(round(pd_region_vec_nonzero.numel()*D_01_idx), dim=0, largest=False).values[-1]
                mean_pd = pd_region_vec_nonzero.mean()

                # ground truth dose
                gt_region_vec = gt_vec * mask_vec
                gt_region_vec_nonzero = gt_region_vec[gt_region_vec.nonzero(as_tuple = False).detach()]
                if gt_region_vec_nonzero.numel() == 0:
                    continue
                D_01_gt = gt_region_vec_nonzero.topk(round(gt_region_vec_nonzero.numel()*D_01_idx), dim=0, largest=False).values[-1]
                mean_gt = gt_region_vec_nonzero.mean()

                # compute diff
                D_01_diff += torch.abs(D_01_pd - D_01_gt)
                mean_diff += torch.abs(mean_pd - mean_gt)
                n         += 2

            # ptv masks need to calculate D_99, D_95 and D_1
            if roi >= 7:
                
                # idx is the location of intensity value
                length = mask_vec.nonzero(as_tuple = False).numel()
                D_99_idx = round(length * 0.01)
                D_95_idx = round(length * 0.05)
                D_1_idx = round(length * 0.99)

                # pred dose
                pd_region_vec = pd_vec * mask_vec
                pd_region_vec_nonzero = pd_region_vec[pd_region_vec.nonzero(as_tuple = False).detach()]
                if pd_region_vec_nonzero.numel() == 0:
                    continue
                try:
                    D_99_pd = pd_region_vec_nonzero.topk(round(pd_region_vec_nonzero.numel()*0.01), dim=0, largest=False).values[-1]
                except IndexError:
                    D_99_pd = pd_region_vec_nonzero.topk(round(pd_region_vec_nonzero.numel()*0.01)+1, dim=0, largest=False).values[-1]

                D_95_pd = pd_region_vec_nonzero.topk(round(pd_region_vec_nonzero.numel()*0.05), dim=0, largest=False).values[-1]
                D_1_pd  = pd_region_vec_nonzero.topk(round(pd_region_vec_nonzero.numel()*0.99),  dim=0, largest=False).values[-1]

                # ground truth dose
                gt_region_vec = gt_vec * mask_vec
                gt_region_vec_nonzero = gt_region_vec[gt_region_vec.nonzero(as_tuple = False).detach()]
                if gt_region_vec_nonzero.numel() == 0:
                    continue
                try:
                    D_99_gt = pd_region_vec_nonzero.topk(round(gt_region_vec_nonzero.numel()*0.01), dim=0, largest=False).values[-1]
                except IndexError:
                    D_99_gt = pd_region_vec_nonzero.topk(round(gt_region_vec_nonzero.numel()*0.01)+1, dim=0, largest=False).values[-1]
                D_95_gt = gt_region_vec_nonzero.topk(round(gt_region_vec_nonzero.numel()*0.05), dim=0, largest=False).values[-1]
                D_1_gt  = gt_region_vec_nonzero.topk(round(gt_region_vec_nonzero.numel()*0.99),  dim=0, largest=False).values[-1]

                # compute diff
                D_99_diff += torch.abs(D_99_pd - D_99_gt)
                D_95_diff += torch.abs(D_95_pd - D_95_gt)
                D_1_diff  += torch.abs(D_1_pd  - D_1_gt)
                n         += 3

    return (D_01_diff + mean_diff + D_99_diff + D_95_diff + D_1_diff) / n
  
  
