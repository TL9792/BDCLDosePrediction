import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.utils.data.dataset import ConcatDataset


def dvh_loss_v(pd, gt, masks):
    """
    masks should have 10 roi masks including
    oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
            'Esophagus', 'Larynx', 'Mandible'], targets=['PTV56', 'PTV63', 'PTV70']
    """

    # initialize the diff
    Loss = 0
    n = 0
    max_w = 0
    H = torch.tensor(pd.shape[2])
    W = torch.tensor(pd.shape[3])
    C = torch.tensor(pd.shape[4])

    # get batch size and roi masks num
    batch_size = len(pd)
    roi_num = len(masks[0, ...])
    for batch in range(batch_size):
        for roi in range(roi_num):
            
            # first flatten (128,128,128)
            pd_vec = pd[batch, 0, ...].flatten()
            gt_vec = gt[batch, 0, ...].flatten()
            mask_vec = masks[batch, roi, ...].flatten().type(torch.bool)

            # if mask has no value e.g. all zero, then next roi
            if mask_vec.nonzero(as_tuple = False).numel() == 0:
                continue
                
            # get mask val in pred
            pd_nonzero_val = pd_vec[mask_vec]

            # sort the mask val
            pd_sort_val = pd_nonzero_val.topk(pd_nonzero_val.numel(), dim=0).values

            # get mask idx and mask val in gt
            gt_nonzero_val = gt_vec[mask_vec]

            # sort the mask val
            gt_sort_val = gt_nonzero_val.topk(gt_nonzero_val.numel(), dim=0).values

            # calculate the value-based DVH loss
            Loss += (torch.abs(gt_sort_val - pd_sort_val)).sum()

            # the total voxel nums in ROIs 
            n += pd_sort_val.numel()

    return (Loss / (n))

