#This code is used to remove a variable n from the dataset

import torch
import numpy as np

def removeVariable(series, masks, diffs, n):
    series, masks, diffs = torch.transpose(series,-1,0),torch.transpose(masks,-1,0), torch.transpose(diffs,-1,0)
    series = torch.cat([series[0:n], series[n+1:]])
    masks = torch.cat([masks[0:n],masks[n+1:]])
    diffs = torch.cat([diffs[0:n],diffs[n+1:]])
    return torch.transpose(series,-1,0), torch.transpose(masks,-1,0), torch.transpose(diffs,-1,0)
