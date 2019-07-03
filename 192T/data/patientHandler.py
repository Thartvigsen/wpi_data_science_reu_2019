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
def handleZeros(realPatient, numTimeSteps):

    return torch.from_numpy(np.zeros(numTimeSteps))
