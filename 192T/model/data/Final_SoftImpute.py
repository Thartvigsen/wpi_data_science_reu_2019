#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import importlib
from fancyimpute import SoftImpute, BiScaler

def softImpute(timeSeries, masks, numPatients, numTimeSteps, numVars):

     print("Hello")
 
     numpyTimeSeries = np.asarray(timeSeries, dtype = np.float64)
     numpyMasks = np.asarray(masks, dtype = np.float64)
 
     for i in range(numPatients):
 
         # doing softImpute on one patient at a time, "i" times     shape: 192 x 59

 
         for j in range(numVars):
             if(j<=13):
                 continue
            
             if (int(sum(numpyTimeSeries[i, ..., j])) == numTimeSteps)
                 continue
 
             for y in range(numTimeSteps):
                 
                 if (numpyMasks[i,y,j]) == 1:
                     numpyTimeSeries[i, y, j] = None

             
             # doing softImpute on one patient at a time, "i" times     shape: 192 x 59
    
         patientTS = SoftImpute().fit_transform(numpyTimeSeries[i])
 
             # stores into one giant tensor, 6261 x 192 x 59
         timeSeries[i, ..., ...] = torch.from_numpy(np.asarray(patientTS))
         
         if i == 61:
             print('series for patient ', i , ' variables 0-5: ', timeSeries[i, ..., 0:5])
             print('masks for patient ', i , ' variables 0-5: ', numpyMasks[i, ..., 0:5])
             print()
             print('series for patient ', i , ' variables 20-25: ', timeSeries[i, ..., 20:25])
             print('masks for patient ', i , ' variables 20-25: ', numpyMasks[i, ..., 20:25])
             print()
             print('series for patient ', i , ' variables 54-58: ', timeSeries[i, ..., 54:58])
             print('masks for patient ', i , ' variables 54-58: ', numpyMasks[i, ..., 54:58])
     
     torch.save(timeSeries, 'soft_data.pt')
     print("Saved!")



