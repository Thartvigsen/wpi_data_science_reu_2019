#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from fancyimpute import KNN

def KNNImpute(seriesTensor, masksTensor, numPatients, numTimeSteps, numVars)

    series = np.asarray(timeSeries, dtype = np.float64)
    masks = np.asarray(masks, dtype = np.float64)
        
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
            print('series for patient ', i ', variables 0-5: ', timeSeries[i, ..., 0:5])
            print('masks for patient ', i ', variables 0-5: ', masks[i, ..., 0:5])
            print()
            print('series for patient ', i ', variables 20-25: ', timeSeries[i, ..., 20:25])
            print('masks for patient ', i ', variables 20-25: ', masks[i, ..., 20:25])
            print()
            print('series for patient ', i ', variables 54-58: ', timeSeries[i, ..., 54:58])
            print('masks for patient ', i ', variables 54-58: ', masks[i, ..., 54:58])
 
     torch.save(timeSeries, 'KNN_time_series.pt')
     print("Saved!")

