#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import importlib
import matplotlib.pyplot as plt
from fancyimpute import SoftImpute, BiScaler

def softImpute(timeSeries, masks, numPatients, numTimeSteps, numVars):
 
     numpyTimeSeries = np.asarray(timeSeries, dtype = np.float64)
     numpyMasks = np.asarray(masks, dtype = np.float64)
 
     for i in range(numPatients):
 
         # doing softImpute on one patient at a time, "i" times     shape: 192 x 59
 
         patientTS = SoftImpute().fit_transform(numpyTimeSeries[i])
 
         for j in range(numVars):
             
             """
             rawPatientColumnTS, rawPatientColumnMask = getPatient(timeSeries, masks, i, j)
 
             # realPatient is an array (<=192 x 1)  with only OBSERVED values  
             realPatient = patientReal(rawPatientColumnTS, rawPatientColumnMask, numTimeSteps)
 
             # if there are no observed values, a.k.a all values in the column are 0
             if (len(realPatient) == 0):
                 globalMean = handleZeros(j, timeSeries, diffs)
                 timeSeries[i,..., j] = globalMean
                 print(timeSeries[i,...,j])
 
             else:
             """
             # shape: (192, ) (just one column)
             oneTimeSeries = np.asarray(patientTS[:, j]) # for patient i and variable j, take the column
 
             # stores into one giant tensor, 6261 x 192 x 59
             timeSeries[i, ..., j] = torch.from_numpy((np.asarray(oneTimeSeries)))
 
     
     print(timeSeries)
     torch.save(timeSeries, 'soft_time_series.pt')
     print("Saved!")



