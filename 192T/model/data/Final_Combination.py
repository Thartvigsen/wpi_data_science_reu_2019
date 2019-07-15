#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import importlib
import matplotlib.pyplot as plt

def simpleCombination(TSarray, maskArray):
    nColumns = len(TSarray[0]) # len: 3
    nRows = len(TSarray) # len: 12
    
    for col in range(nColumns): # 3
        row = 0
        while(row < nRows): # 12
                                    
            # if the variable is the first row and is missing, keep looking one row ahead until
            # you have found first non-missing value and impute the first row with THAT value
            if (row==0) and maskArray[row][col] == 1:
                subIndex = 1
                while subIndex < nRows:
                    if maskArray[subIndex][col] == 0:
                        TSarray[row][col] = TSarray[subIndex][col]
                        break
                    subIndex +=1
                row+=1

            # else, if the missing variable is anywhere else in the list, search up and down the column for the 
            # observed values. say the column is [0, 5, 0, 4, 0, 0, 3, 0]. needs to turn into [5, 5, 4.5, 4, 4, 3, 3, 3]. 
            # if index of missing <= (low + high) / 2, impute with low. Else, impute with high. 
            
            elif(maskArray[row][col] == 0): # if a value is observed

                earlyValue = TSarray[row][col]
                subIndex = row + 1
                nZeros = 0
                                
                rowCopy = row

                # search for the next observed value (mask = 0)

                while subIndex <= nRows:
                    
                    
                    if(subIndex==nRows):
                        TSarray = np.transpose(TSarray)
                        TSarray[col][rowCopy-1:subIndex] = earlyValue
                        TSarray = np.transpose(TSarray)


                    elif maskArray[subIndex][col] == 0:

                        lateValue = TSarray[subIndex][col]
                        midIndex = (int)((rowCopy+1 + subIndex) / 2)
                        
                        # if the number of missing values between the two observed is odd
                        if (nZeros > 0 and nZeros % 2 == 1): # need to check that number of zeroes > 0 in case of something like [4, 2, 0, 0]
                            TSarray = np.transpose(TSarray)
                            TSarray[col][midIndex] = (earlyValue + lateValue) / 2.0
                            TSarray[col][rowCopy+1: midIndex] = earlyValue
                            TSarray[col][(midIndex+1): subIndex+1] = lateValue
                            TSarray = np.transpose(TSarray)
                            row+=1
                            break
                        elif nZeros > 0: 
                            # if the number of missing values between the two observed is even
                            TSarray = np.transpose(TSarray)
                            TSarray[col][rowCopy+1 : midIndex+1] = earlyValue
                            TSarray[col][midIndex: subIndex+1] = lateValue
                            TSarray = np.transpose(TSarray)
                            row+=1
                            break
                    
                               
                    nZeros += 1
                    subIndex += 1
                    row+=1
                                         
            elif (maskArray[row][col] == 1):
                TSarray[row][col] = TSarray[row-1][col]
                row+=1
                                                
    return TSarray, maskArray


def tensorCombination(timeSeries, masks, numPatients, numTimeSteps, numVars):
    
    numpyTimeSeries = timeSeries.numpy()
    numpyMasks = masks.numpy()

    for i in range(numPatients):

       # doing LOCF on one patient at a time, "i" times     shape: 192 x 59

       patientTS, patientMask = simpleCombination(numpyTimeSeries[i], numpyMasks[i])

       for j in range(numVars):
          """ 
          # the only reason we make raw patient columns is to pass into "patientReal", which generates array with only observed values
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
    torch.save(timeSeries, 'for_back_combination_time_series.pt')
    print("Saved!")

