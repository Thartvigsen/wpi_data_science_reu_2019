from data.meanImpute import meanImpute
from dataCleaner import get_sizes

def performImpute(dataObj):

    numPatients, numTimeSteps, numVars = get_sizes(dataObj.data)
    meanImpute(dataObj.data, dataObj.masks, numPatients, numTimeSteps, numVars)

int main()
