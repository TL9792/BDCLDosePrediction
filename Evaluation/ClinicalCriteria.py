import numpy as np
import torch

def cal_CI(pd,gt,roi):
  """
  pd is the predicted dose array, and gt is the ground truth dose array
  roi represents roi masks including three PTV masks (56Gy, 63Gy, 70Gy) and seven OAR masks (Brainstem, SpinalCord, RightParotid, LeftParotid,
            Esophagus, Larynx, Mandible), whose dimention is (10,128,128,128)
  """
  
  
  
  
