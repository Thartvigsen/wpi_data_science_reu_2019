#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np
#import importlib
#import matplotlib.pyplot as plt
from data.patientHandler import patientReal, getPatient, handleZeros


def simpleLOCF(TSarray, maskArray):
            
    # Now, we actually do LOCF
    
    rowLength = len(TSarray[0]) # len: 3
    columnLength = len(TSarray) # len: 7
    
    for col in range(rowLength): # 3
        for row in range(columnLength): # 7
            
            # if the variable is the first row and is missing, keep looking one row ahead until
            # you have found first non-missing value
            if (row==0) and maskArray[row][col] == 1:
                subIndex = 1
                while subIndex < columnLength:
                    if maskArray[subIndex][col] == 0:
                        TSarray[row][col] = TSarray[subIndex][col]
                        break
                    subIndex +=1

            # else, if the missing variable is anywhere else in the list
            elif(maskArray[row][col] == 1):
                TSarray[row][col] = TSarray[row-1][col]
            
    return TSarray, maskArray


# this does forward imputation for the big data
# INPUT NUMBER OF PATIENTS, TIMESERIES TENSOR, MASKS TENSOR AND DIFFS

def tensorLOCF(timeSeries, masks, numPatients, numTimeSteps, numVars):
    
    numpyTimeSeries = timeSeries.numpy()
    numpyMasks = masks.numpy()

    for i in range(numPatients):
        
        # doing LOCF on one patient at a time, "i" times     shape: 192 x 59
	
        patientTS, patientMask = simpleLOCF(numpyTimeSeries[i], numpyMasks[i])      

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
    torch.save(timeSeries, 'forward_time_series.pt')
    print("Saved!") 
