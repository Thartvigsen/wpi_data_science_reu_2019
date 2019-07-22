import torch
import numpy as np
#from patientHandler import patientReal

def patientReal(oneSeries, oneMask, numTimeSteps):
    realSeries=[]
    for k in range(numTimeSteps):
        if (int(oneMask[k])==0):
            realSeries.append(oneSeries[k])
    return realSeries

def meanImpute(seriesTensor, masksTensor, numPatients, numTimeSteps, numVars):

    series = np.asarray(seriesTensor, dtype = np.float64)
    masks = np.asarray(masksTensor, dtype = np.float64)
    print('numVar: ', numVars)
    
    for i in range(numPatients):
        for j in range(numVars):
            oneRealSeries = patientReal(series[i, : , j], masks[i,:, j], numTimeSteps)
            if len(oneRealSeries)==0:
                pass
            else:
                mean = np.mean(oneRealSeries)
                for y in range(numTimeSteps):
                    if int(masks[i,y,j])==1:
                        series[i,y,j]=mean
            seriesTensor[i, : , j]=torch.from_numpy(series[i,:,j])
        
        if i == 1:
            print('series for patient ', i )
            print('series for variables 0-5: ', seriesTensor[i, ..., 0:5])
            print('masks for variables 0-5: ', masksTensor[i, ..., 0:5])
            print()
            print('series for variables 20-25: ', seriesTensor[i, ..., 20:25])
            print('masks for variables 20-25: ', masksTensor[i, ..., 20:25])
            print()
            print('series for variables 54-58: ', seriesTensor[i, ..., 54:58])
            print('masks for variables 54-58: ', masksTensor[i, ..., 54:58])
 
    torch.save(seriesTensor, 'mean_time_series.pt')
    print("Saved!")


# In[ ]:




