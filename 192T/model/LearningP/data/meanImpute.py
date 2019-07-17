'''
This function imputes mean values for missing data. It calculates the mean for a patient in a variable and then propogates that through the missing values.
'''

import numpy as np
import torch
from data.patientHandler import patientReal, getPatient

def meanImpute(data, masks, numPatients, numTimeSteps, numVariables):

    for i in range(numPatients): #Go through each patient
        for j in range(numVariables): #Go through each variable 

            patient, patientMasks = getPatient(data, masks, i, j) #Get time steps for variable

            realPatient = patientReal(patient, patientMasks, numTimeSteps) # Get array of patient with only real data
        
            #Insert the mean for any missing values
            
            if(np.sum(realPatient)==0):
               continue

            mean = np.mean(realPatient)

            for k in range(numTimeSteps): #Iterate through time steps

                if(patientMasks[k]==1): #Check if value is missing

                    realPatient.insert(k, mean) #Impute mean if missing value

            data[i, ..., j] = torch.from_numpy(np.asarray(realPatient)) #Set data for variable

    torch.save(data,'mean_time_series.pt')
    print("Saved!") 
