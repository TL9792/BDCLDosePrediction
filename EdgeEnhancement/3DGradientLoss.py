import torch

def gradient_loss(pd, gt, mask):
  "pd is the predicted dose distribution map, gt represents the ground truth and mask represents the dose mask."
  
    sobelx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    sobely = torch.tensor([ [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], 
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    sobelz = torch.tensor([ [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
                            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).expand(pd.shape[0], 1, 3, 3, 3).cuda().float()
    pdx = F.conv3d(pd.float(), torch.cat([sobelx]*9, dim=1), padding=1, stride=1)
    gtx = F.conv3d(gt.float(), torch.cat([sobelx]*9, dim=1), padding=1, stride=1)
    pdy = F.conv3d(pd.float(), torch.cat([sobely]*9, dim=1), padding=1, stride=1)
    gty = F.conv3d(gt.float(), torch.cat([sobely]*9, dim=1), padding=1, stride=1)
    pdz = F.conv3d(pd.float(), torch.cat([sobelz]*9, dim=1), padding=1, stride=1)
    gtz = F.conv3d(gt.float(), torch.cat([sobelz]*9, dim=1), padding=1, stride=1)        
    return ( torch.abs(pdx - gtx).sum() + torch.abs(pdy - gty).sum() + torch.abs(pdz - gtz).sum() ) / mask.sum()
  
  
