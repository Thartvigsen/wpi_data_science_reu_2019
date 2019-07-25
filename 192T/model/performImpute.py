<<<<<<< HEAD
from data.Final_PCAImpute import MissForestImpute
=======
from data.Final_KNNImpute import KNNImpute
>>>>>>> 39e14436a36b8b7a34e2fea499d251c6469d0705
from dataCleaner import get_sizes
from data.dataLoader import dataLoader

def main():

    print("Hello there")

    dataObj = dataLoader("data.pt", "labels.pt", "masks.pt", "diffs.pt", 32) # Load in the data

    performImpute(dataObj)

    print("Impute completed!")


def performImpute(dataObj):

    numPatients, numTimeSteps, numVars = get_sizes(dataObj.data)

    KNNImpute(dataObj.data, dataObj.masks, numPatients, numTimeSteps, numVars)  

main()
