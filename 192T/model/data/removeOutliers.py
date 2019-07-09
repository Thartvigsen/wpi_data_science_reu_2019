'''
This method removes any outliers from the data. Outliers are defined as data two or more standard deviations away. Replaces outliers with mean. Represent all outliers as missing in masks tensor.
'''

import torch
import numpy as np

def removeOutliers(TS, masks, diffs, numPatients, numTimeSteps, numVariables):
    TS = np.array(TS) 
    finalTS = []
    for n in range(numPatients):
        cleanTS = []
        for i in range(numVariables):
            cleanVar = []
            var = TS[n,:,i]
        
            mean = np.mean(var)
            std = np.std(var)
            boundSet = (mean - 2 * std, mean + 2 * std)
            for j in range(numTimeSteps):
                y = var[j]
                if(y>=boundSet[0] and y <= boundSet[1]):
                    cleanVar.append(y)
                else:
                    cleanVar.append(0)
                    masks[n][j][i] = 1
                    if(j==0):
                        diffs[n][j][i] = 0
                    else:
                        diffs[n][j][i] = diffs[n][j-1][i] + 1/48
            cleanTS.append(cleanVar)

        cleanTS = np.asarray(cleanTS)
        finalTS.append(cleanTS)
    finalTS = torch.from_numpy(np.asarray(finalTS))
    finalTS = torch.transpose(finalTS, 1, 2)
    return finalTS, masks, diffs
        
    
