import torch
import numpy as np
from fancyimpute import KNN

def KNNImpute(seriesTensor, masksTensor, numPatients, numTimeSteps, numVars):

    series = np.asarray(seriesTensor, dtype = np.float64)
    masks = np.asarray(masksTensor, dtype = np.float64)
    seriesToOutput = torch.zeros([numPatients, numTimeSteps, numVars], dtype=torch.float32)
    
    for i in range(numPatients):
        for j in range(numVars):
            if (sum(series[i,...,j])==numTimeSteps):
                #seriesToOutput[i,...,j]=seriesTensor[i,...,j]
                continue
            for y in range(numTimeSteps):
                if int(masks[i,y,j])==1:
                    series[i, y, j]=None
                    
        #per patient, perform imputation##
        filledPatientTS = KNN(k=3).fit_transform(series[i])
        
        #putting time series vectors in original tensor shapes
        seriesToOutput[i, ..., ...] = torch.from_numpy(filledPatientTS)
        
        if i == 1:
            print('patient ', i )
            print('series for variables 0-5: ', seriesToOutput[i, ..., 0:5])
            print('masks for variables 0-5: ', masksTensor[i, ..., 0:5])
            print()
            print('series for variables 20-25: ', seriesToOutput[i, ..., 20:25])
            print('masks for variables 20-25: ', masksTensor[i, ..., 20:25])
            print()
            print('series for variables 54-58: ', seriesToOutput[i, ..., 54:58])
            print('masks for variables 54-58: ', masksTensor[i, ..., 54:58])
 
    torch.save(seriesToOutput, 'KNN_time_series.pt')
    print("Saved!")

