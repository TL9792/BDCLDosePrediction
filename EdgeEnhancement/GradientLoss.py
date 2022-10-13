import torch
import torch.nn.functional as F 


def gradient_loss(pd, gt, mask):
    """
    pd is the predicted dose distribution map, gt represents the ground truth, and mask represents the dose mask
    pd: b, 1, 128, 128, 128
    gt: b, 1, 128, 128, 128
    mask: b, 1, 128, 128, 128
    """
    
    sobelx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    sobely = torch.tensor([ [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], 
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    sobelz = torch.tensor([ [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
                            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    pdx = F.conv3d(pd.float(), sobelx, padding=1, stride=1)
    gtx = F.conv3d(gt.float(), sobelx, padding=1, stride=1)
    pdy = F.conv3d(pd.float(), sobely, padding=1, stride=1)
    gty = F.conv3d(gt.float(), sobely, padding=1, stride=1)
    pdz = F.conv3d(pd.float(), sobelz, padding=1, stride=1)
    gtz = F.conv3d(gt.float(), sobelz, padding=1, stride=1)        
    return ( torch.abs(pdx - gtx).sum() + torch.abs(pdy - gty).sum() + torch.abs(pdz - gtz).sum() ) / mask.sum()

  
  
