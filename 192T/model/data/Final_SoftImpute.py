#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import importlib
from fancyimpute import SoftImpute, BiScaler

def softImpute(timeSeries, masks, numPatients, numTimeSteps, numVars):
 
     numpyTimeSeries = np.asarray(timeSeries, dtype = np.float64)
     numpyMasks = np.asarray(masks, dtype = np.float64)
 
     for i in range(numPatients): 
 
         for j in range(numVars):
             
             for y in range(numTimeSteps):
                 
                 if (numpyMasks[i,y,j]) == 1:
                     numpyTimeSeries[i, y, j] = None

             
         # doing softImpute on one patient at a time, "i" times     shape: 192 x 59
    
         patientTS = SoftImpute().fit_transform(numpyTimeSeries[i])

         # shape: (192, ) (just one column)
         oneTimeSeries = np.asarray(patientTS[:, j]) # for patient i and variable j, take the column
 
         # stores into one giant tensor, 6261 x 192 x 59
         timeSeries[i, ..., j] = torch.from_numpy((np.asarray(oneTimeSeries)))
         
         if i == 1:
             print('series for patient ', i , ' variables 0-5: ', timeSeries[i, ..., 0:5])
             print('masks for patient ', i , ' variables 0-5: ', numpyMasks[i, ..., 0:5])
             print()
             print('series for patient ', i , ' variables 20-25: ', timeSeries[i, ..., 20:25])
             print('masks for patient ', i , ' variables 20-25: ', numpyMasks[i, ..., 20:25])
             print()
             print('series for patient ', i , ' variables 54-58: ', timeSeries[i, ..., 54:58])
             print('masks for patient ', i , ' variables 54-58: ', numpyMasks[i, ..., 54:58])
     
     print(timeSeries)
     torch.save(timeSeries, 'soft_time_series.pt')
     print("Saved!")



