#This code splits the data into training, testing, and validation sets

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def getSplits(dataObj, train, validation, test):
    split_props = [train, validation, test]
    indices = range(len(dataObj.data))
    split_points = [int(len(dataObj.data)*i) for i in split_props]
    train_ix = np.random.choice(indices, split_points[0], replace=False)
    validation_ix = np.random.choice(list(set(indices) - set(train_ix)), split_points[1], replace=False)
    test_ix = np.random.choice(list(set(indices)-set(validation_ix)-set(train_ix)), split_points[2], replace=False)

    
    return train_ix, validation_ix, test_ix

def calculateSplits(dataObj, train, validation, test, BATCH_SIZE):

    train_ix, validation_ix, test_ix = getSplits(dataObj, train, validation, test)
    
    train_sampler = SubsetRandomSampler(train_ix)
    validation_sampler = SubsetRandomSampler(validation_ix)
    test_sampler = SubsetRandomSampler(test_ix)

    train_loader = torch.utils.data.DataLoader(dataset = dataObj, batch_size = BATCH_SIZE, sampler = train_sampler, shuffle = False, drop_last = True)
    
    validation_loader = torch.utils.data.DataLoader(dataset = dataObj, batch_size = BATCH_SIZE, sampler = validation_sampler, shuffle = False, drop_last = True)

    test_loader = torch.utils.data.DataLoader(dataset = dataObj, batch_size = BATCH_SIZE, sampler = test_sampler, shuffle = False, drop_last = True)

    return train_loader, validation_loader, test_loader
