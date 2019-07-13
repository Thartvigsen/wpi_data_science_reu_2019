from data.meanImpute import meanImpute
from dataCleaner import get_sizes
from data.dataLoader import dataLoader

dataObj = dataLoader("time_series.pt", "labels.pt", "masks.pt", "diffs.pt", 32) # Load in the data

performImpute(dataObj)

print("Impute completed!")


def performImpute(dataObj):

    numPatients, numTimeSteps, numVars = get_sizes(dataObj.data)
    meanImpute(dataObj.data, dataObj.masks, numPatients, numTimeSteps, numVars)  

