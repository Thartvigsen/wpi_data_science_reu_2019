'''
This class loads in data provided by the user. It stores the data and accompanying labels as instance variables. It can call a variety imputation methods used to fill in missing values
'''

import torch
import torch.utils.data
from data.Final_KNNImpute import KNNImpute
import numpy as np
#from data.removeOutliers import removeOutliers
from data.calculateSplits import calculateSplits
from data.removeVariable import removeVariable
from dataCleaner import get_sizes
from data.catTorch import addMasks

class dataLoader(torch.utils.data.Dataset):


    def __init__(self, series, labels, masks, diffs, BATCH_SIZE):
        super(dataLoader, self).__init__()
        self.data, self.masks, self.labels, self.diffs = self.load_medical(series, labels, masks, diffs, 6261) #Generate the data
        #self.data = addMasks(self.data, self.masks)
        self.train_ix, self.validation_ix, self.test_ix = calculateSplits(self, .8, .1, .1, BATCH_SIZE) #Split the data into train, validation, test

    def __getitem__(self, index):
       return self.data[index], self.labels[index]


    def __len__(self):
        return len(self.data)


    #Take a subset of the data
    def spliceData(self, series, masks, labels, diffs, n):

        return(series[:n], masks[:n], labels[:n], diffs[:n])


    #Load the data, including splicing and removing outliers
    def load_medical(self, series, labels, masks, diffs, splice):
        series, masks, labels, diffs = torch.load(series), torch.load(masks), torch.load(labels).squeeze(), torch.load(diffs)
 
        series, masks, labels, diffs = self.spliceData(series, masks, labels, diffs, splice)

        numPatients, numTimeSteps, numVars = get_sizes(series)
        
        return series.type(torch.FloatTensor), masks, labels, diffs
