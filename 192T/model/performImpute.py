from data.Final_SoftImpute import softImpute
from dataCleaner import get_sizes
from data.dataLoader import dataLoader

def main():

    print("Hello there")

    dataObj = dataLoader("time_series.pt", "labels.pt", "masks.pt", "diffs.pt", 32) # Load in the data

    performImpute(dataObj)

    print("Impute completed!")


def performImpute(dataObj):

    numPatients, numTimeSteps, numVars = get_sizes(dataObj.data)
    softImpute(dataObj.data, dataObj.masks, numPatients, numTimeSteps, numVars)  

main()
