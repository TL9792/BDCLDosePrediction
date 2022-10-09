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

    # initial of the diff
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

            # if mask has no value eg. all zero, then next roi
            if mask_vec.nonzero(as_tuple = False).numel() == 0:
                continue
                
            # get mask idx and mask val in pred
            pd_nonzero_val = pd_vec[mask_vec]
            pd_nonzero_idx = mask_vec.nonzero(as_tuple = False)

            # sort the mask val
            pd_sort_val = pd_nonzero_val.topk(pd_nonzero_val.numel(), dim=0).values
            pd_sort_idx = pd_nonzero_val.topk(pd_nonzero_val.numel(), dim=0).indices

            # get the mask val indx in pred image
            pd_ori_idx = pd_nonzero_idx[pd_sort_idx]


            # get mask idx and mask val in gt
            gt_nonzero_val = gt_vec[mask_vec]
            gt_nonzero_idx = mask_vec.nonzero(as_tuple = False)

            # sort the mask val
            gt_sort_val = gt_nonzero_val.topk(gt_nonzero_val.numel(), dim=0).values
            gt_sort_idx = gt_nonzero_val.topk(gt_nonzero_val.numel(), dim=0).indices

            # get the mask val indx in gt image
            gt_ori_idx = gt_nonzero_idx[gt_sort_idx]

            Loss += ( torch.abs(gt_sort_val - pd_sort_val)).sum()

            n += pd_sort_val.numel()

    return (Loss / (n))
