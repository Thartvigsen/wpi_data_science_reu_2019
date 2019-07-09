#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code deals with raw patient data
'''

import torch
import numpy as np

#Create a patient array with only real values
def patientReal(patient, patientMasks, numTimeSteps):

    realPatient = []
    for k in range(numTimeSteps):
        if(patientMasks[k]==0):
            realPatient.append(k)
    return realPatient

#Get patient and patientMasks for specific variable
def getPatient(data, masks, i, j):

    patient = np.asarray(data[i,...,j])
    patientMasks = np.asarray(masks[i,...,j])

    return patient, patientMasks

#This function defines what to do if all values for a variable are 0
def handleZeros(variableNum, allTimeSeries, allDiffs):           #allTimeSeries, AllDiffs shape = (6261, 192, 58)
    meanForPatientArray=[]
    
    numOfPatients=len(allTimeSeries[...,1,1])
    
    for i in range(numOfPatients):
        timeSeriesForPatientAndVariable = np.asarray(allTimeSeries[i, ... , variableNum])
        diffForPatientAndVariable = np.asarray(allDiffs[i, ..., variableNum])
        if (diffForPatientAndVariable[-1] >=  ( (1/48) * (len(allTimeSeries[1,...,1])-1) ) ):
            #if all values are missing (last diffs value is > threshhold corresponding to no observed values)
            pass
        else:
            #there's at least one observation
            meanForPatient = np.mean(timeSeriesForPatientAndVariable, dtype ='float64')
            meanForPatientArray.append(meanForPatient)
    
    
    #CHANGE BOUNDS BELOW
    if(len(meanForPatientArray)<(.05*numOfPatients)):     #if there are too few observations for this variable
        globalMean = 0.0                        
        #impute with zero because there's too few patients tested for this variable
        #healthy ppl are not tested for this variable
    else:
        globalMean = np.mean(meanForPatientArray, dtype ='float64')
    
    return globalMean


# In[ ]:




