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

    realSeries, realMask = [], []
    for k in range(numTimeSteps):
        if(patientMasks[k]==0):
            realSeries.append(patient[k])
            realMask.append(k)
    return realSeries, realMask

#Get patient and patientMasks for specific variable
def getPatient(data, masks, i, j):

    patient = np.asarray(data[i,...,j])
    patientMasks = np.asarray(masks[i,...,j])

    return patient, patientMasks



