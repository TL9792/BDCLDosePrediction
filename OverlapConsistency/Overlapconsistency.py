def consistloss(pd,msk):
    """
    pd is the predicted beam voters, and msk represents beam masks
    pd: b, 9, 128, 128, 128
    msk: b, 9, 128, 128, 128
    """
    
    sum_pd = torch.sum(pd, dim=1).unsqueeze(1)
    sum_msk = torch.sum(msk, dim=1).unsqueeze(1)

    sum_msk[sum_msk < 2] = 1   # (2,1,128,128,128)
    overlap_sum_msk = sum_msk > 1
    overlap_msk = overlap_sum_msk * msk
    avg = (sum_pd*overlap_sum_msk) / sum_msk
    diff = torch.abs(pd * overlap_msk - avg)
    loss = (torch.sum(diff) / torch.sum(overlap_msk))

    return loss
  
