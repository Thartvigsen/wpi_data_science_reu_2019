'''
This class loads in data provided by the user. It stores the data and accompanying labels as instance variables. It can call a variety imputation methods used to fill in missing values
'''

import torch
import torch.utils.data
from meanImpute import meanImpute
import numpy as np
from removeOutliers import removeOutliers
from calculateSplits import calculateSplits
from removeData import removeData

class dataLoader(torch.utils.data.Dataset):


    def __init__(self, series, labels, masks, BATCH_SIZE):
        super(dataLoader, self).__init__()
        self.data, self.labels = self.generate_data(series, labels, masks) #Generate the data
        self.train_ix, self.validation_ix, self.test_ix = calculateSplits(self, .8, .1, .1, BATCH_SIZE) #Split the data into train, validation, test




    #Load the data and perform imputation. This is the only part of the code to change!!!
    def generate_data(self, series, labels, masks):

        series, masks, labels, numPatients, numTimeSteps, numVars = self.load_medical(series, labels, masks, 500)

        series, masks = meanImpute(series, masks, numPatients, numTimeSteps, numVars) #Use mean impute
        return series.type(torch.FloatTensor), labels 



    def __getitem__(self, index):
       return self.data[index], self.labels[index]


    def __len__(self):
        return len(self.data)


    #Get and return the three dimensions of the time series - patients x timeSteps x vars
    def get_sizes(self, series):
        sizeArr = list(series.size())
        return(sizeArr[0], sizeArr[1], sizeArr[2])

    #Take a subset of the data
    def spliceData(self, series, masks, labels, n):

        return(series[:n], masks[:n], labels[:n])


    #Load the data, including splicing and removing outliers
    def load_medical(self, series, labels, masks, splice):
        series, masks, labels = torch.load(series), torch.load(masks), torch.load(labels).squeeze()
        series, masks = removeData(series, masks, 19)
        series, masks, labels = self.spliceData(series, masks, labels, splice)
        
        numPatients, numTimeSteps, numVars = self.get_sizes(series)
        series, masks = removeOutliers(series, masks, numPatients, numTimeSteps, numVars)
        return series, masks, labels, numPatients, numTimeSteps, numVars
