import torch

def addMasks(series, masks):

    return(torch.cat((series, masks),2))

def addDiffs(series, diffs):

    return(torch.cat((series, diffs),2))

def addMasksDiffs(series, masks, diffs):

    series = addMasks(series, masks)
    return addDiffs(series, diffs)
