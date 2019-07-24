import torch
import numpy as np
from fancyimpute import KNN

def KNNImpute(seriesTensor, masksTensor, numPatients, numTimeSteps, numVars):

    series = np.asarray(seriesTensor, dtype = np.float64)
    masks = np.asarray(masksTensor, dtype = np.float64)
        
    for i in range(numPatients):
        for j in range(numVars):
            for y in range(numTimeSteps):
                if int(masks[i,y,j])==1:
                    series[i, y, j]=None
                    
        #per patient, perform imputation
        filledPatientTS = KNN(k=3).fit_transform(series[i])
        
        #putting time series vectors in original tensor shapes
        seriesTensor[i, ..., ...] = torch.from_numpy(filledPatientTS)
        
        if i == 1:
            print('patient ', i )
            print('series for variables 0-5: ', seriesTensor[i, ..., 0:5])
            print('masks for variables 0-5: ', masksTensor[i, ..., 0:5])
            print()
            print('series for variables 20-25: ', seriesTensor[i, ..., 20:25])
            print('masks for variables 20-25: ', masksTensor[i, ..., 20:25])
            print()
            print('series for variables 54-58: ', seriesTensor[i, ..., 54:58])
            print('masks for variables 54-58: ', masksTensor[i, ..., 54:58])
 
    torch.save(seriesTensor, 'KNN_time_series.pt')
    print("Saved!")

