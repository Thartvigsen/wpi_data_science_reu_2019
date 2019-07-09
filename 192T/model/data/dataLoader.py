'''
This class loads in data provided by the user. It stores the data and accompanying labels as instance variables. It can call a variety imputation methods used to fill in missing values
'''

import torch
import torch.utils.data
from data.Final_Forward_Imputation import tensorLOCF
import numpy as np
from data.removeOutliers import removeOutliers
from data.calculateSplits import calculateSplits
from data.removeVariable import removeVariable

class dataLoader(torch.utils.data.Dataset):


    def __init__(self, series, labels, masks, diffs, BATCH_SIZE):
        super(dataLoader, self).__init__()
        self.data, self.labels, self.masks, self.diffs = self.generate_data(series, labels, masks, diffs) #Generate the data
        self.train_ix, self.validation_ix, self.test_ix = calculateSplits(self, .8, .1, .1, BATCH_SIZE) #Split the data into train, validation, test




    #Load the data and perform imputation. This is the only part of the code to change!!!
    def generate_data(self, series, labels, masks, diffs):

        series, masks, labels, diffs, numPatients, numTimeSteps, numVars = self.load_medical(series, labels, masks, diffs, 6261)

        series, masks, diffs = NEWMETHOD(series, masks, diffs, numPatients, numTimeSteps, numVars) #Use mean impute
        return series.type(torch.FloatTensor), labels, masks, diffs 


    def __getitem__(self, index):
       return self.data[index], self.labels[index]


    def __len__(self):
        return len(self.data)


    #Get and return the three dimensions of the time series - patients x timeSteps x vars
    def get_sizes(self, series):
        sizeArr = list(series.size())
        return(sizeArr[0], sizeArr[1], sizeArr[2])

    #Take a subset of the data
    def spliceData(self, series, masks, labels, diffs, n):

        return(series[:n], masks[:n], labels[:n], diffs[:n])


    #Load the data, including splicing and removing outliers
    def load_medical(self, series, labels, masks, diffs, splice):
        series, masks, labels, diffs = torch.load(series), torch.load(masks), torch.load(labels).squeeze(), torch.load(diffs)
        series, masks, diffs = removeVariable(series, masks, diffs, 19) #Remove variable 19
        series, masks, labels, diffs = self.spliceData(series, masks, labels, diffs, splice)
        
        numPatients, numTimeSteps, numVars = self.get_sizes(series)
        series, masks, diffs = removeOutliers(series, masks, diffs, numPatients, numTimeSteps, numVars)
        return series, masks, labels, diffs, numPatients, numTimeSteps, numVars
