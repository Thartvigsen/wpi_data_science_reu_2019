import torch
import numpy as np
from data.removeVariable import removeVariable
from data.removeOutliers import removeOutliers

#Get and return the three dimensions of the time series - patients x timeSteps x vars
def get_sizes(series):
    sizeArr = list(series.size())
    return(sizeArr[0], sizeArr[1], sizeArr[2])


def cleanData(series_f, masks_f, diffs_f):

    
    series, masks, diffs = torch.load(series_f), torch.load(masks_f), torch.load(diffs_f)


    series, masks, diffs = removeVariable(series, masks, diffs, 19) #Remove variable 19


    numPatients, numTimeSteps, numVars = get_sizes(series) 
        
    series = handleZeros(series, masks, numPatients, numTimeSteps, numVars) 
    series, masks, diffs = removeOutliers(series, masks, diffs, numPatients, numTimeSteps, numVars)

    torch.save(series, series_f)
    torch.save(masks, masks_f)
    torch.save(diffs, diffs_f)

 
#This function defines what to do if all values for a variable are 0
def handleZeros(data, masks, numPatients, numTimeSteps, numVars):           #allTimeSeries, AllDiffs shape = (6261, 192, 58)
    
    for var in range(numVars):
        print("Starting variable ",var)
        meanForVarArr = []
        for patient in range(numPatients):
            ts = np.asarray(data[patient, ... ,var])
            if (np.sum(np.asarray(masks[patient,...,var]))==numTimeSteps):
                #if all values are missing (last diffs value is > threshhold corresponding to no observed values)
                masks[patient,0,var] = 2
            else:
                #there's at least one observation
                if(np.sum(ts)!=0):
                    meanForPatient = np.mean(ts, dtype ='float64')
                    meanForVarArr.append(meanForPatient)
                else:
                    meanForVarArr.append(0)    
    
        #CHANGE BOUNDS BELOW
        if(len(meanForVarArr)<(.05*numPatients)):     #if there are too few observations for this variable
            globalMean = 0.0                        
            #impute with zero because there's too few patients tested for this variable
            #healthy ppl are not tested for this variable
        else:
            globalMean = np.mean(meanForVarArr, dtype ='float64')
            print(globalMean)
        for patient in range(numPatients):
            if(masks[patient,0,var]==2):
                masks[patient,0,var] = 1
                for step in data[patient,...,var]:
                    step = globalMean
    


    return data

def main():
    cleanData("time_series.pt", "masks.pt", "diffs.pt")
    print("Data cleaned!!!")

if __name__ == "__main__":
    main()

