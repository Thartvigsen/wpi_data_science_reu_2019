
import torch
import numpy as np
from data.patientHandler import patientReal


def normalize(minAllVar, maxAllVar, cleanSeries, masks, numPatients, numTimeSteps):
    normalizedSeries = np.asarray(cleanSeries)
    for var in range(len(minAllVar)):
        print('normalizing - var: ', var)
        minn = minAllVar[var]
        maxx = maxAllVar[var]
        rangg = maxx - minn
        
        for p in range(numPatients):
            for i in range(numTimeSteps):
                y = cleanSeries[p, i, var]
                if (masks[p, i, var]==1.0):
                    pass
                else:
                    normalizedSeries[p, i, var] = (y-minn)/rangg
                                        
    torch.from_numpy(normalizedSeries)
    print('normalized series shape: ', normalizedSeries.shape)
    return normalizedSeries



#following code finds outliers for each patient and medical variable
#based on outlier value and medical variable, it will be decided if outlier should be deleted or not

def removeOutliersNormalize(series, masks, diffs, numPatients, numTimeSteps, numVariables):
    #series, masks, diffs are tensors (6261, 192, 58)
    
    series, masks, diffs = np.asarray(series), np.asarray(masks), np.asarray(diffs)
     
    cleanSeries3DMatrix = []
    minAllVarAllPat, maxAllVarAllPat = [], []
    minAllVar, maxAllVar = [], []
    for v in range(numVariables):
        minAllVarAllPat.append([])
        maxAllVarAllPat.append([])
        minAllVar.append([])
        maxAllVar.append([])
        
    for n in range(numPatients):
        if(n%1000==0):
            print('removing outliers - patient: ', n)
        cleanSeriesPerPatient = []
        for i in range(numVariables):
            cleanSeriesPerVariable = np.zeros(numTimeSteps)
            seriesPerVariable = series[n,:,i]
            if(np.sum(masks[n,:,i])==192):
                cleanSeriesPerVariable = seriesPerVariable
                localMin, localMax = None, None
            else:
                realSeries, realMask = patientReal(seriesPerVariable, masks[n,:,i], numTimeSteps) 
                realSeriesCopy = []
                for elm in realSeries:
                    realSeriesCopy.append(elm)
                if len(realSeriesCopy)==0:
                    localMin, localMax = None, None
                localMin = min(realSeriesCopy)
                localMax = max(realSeriesCopy)
                
                for j in range(len(realSeries)):
                    obs = realSeries[j]
                    realIndex = realMask[j]
                    
                    if(( (i==21 or i==22 or i==28 or i==47) and (obs==0.0) ) 
                       #deals with zero-valued outliers
                        or
                        ( (i==41 and obs==8582.0) or
                          (i==42 and obs==6253.0) or
                          (i==44 and obs==9696.0) or (i==44 and obs==4838.5) or (i==44 and obs==9195.0) or
                          (i==45 and obs==978.0) or
                          (i==47 and obs==920.0) or (i==47 and obs==990.0) or
                          (i==48 and obs==601.0) or
                          (i==49 and obs==901.0) or 
                          (i==50 and obs==835.0) or
                          (i==51 and obs==920.0) or
                          (i==52 and obs==67317.0) or (i==52 and obs==20078.402) or
                          (i==53 and obs==4875.0) or (i==53 and obs==4557.0) or
                          (i==54 and obs==110159.97) or
                          (i==55 and obs==1220.0) or (i==55 and obs==1335.0) or (i==55 and obs==1230.0) or
                          (i==56 and obs==110150.0) or
                          (i==57 and obs==300.0) or (i==57 and obs==390.0) or
                          (i==58 and obs==1000.0) or (i==58 and obs==10085.0)) 
                      ): #deals with selected extreme-valued outliers
                       
                        #print('variable: ', i)
                        #print('outlier value: ', obs)
                        cleanSeriesPerVariable[realIndex]=0.0
                        masks[n][realIndex][i] = 1.0
                        if(realIndex==0):
                            diffs[n][realIndex][i] = 0.0
                        else:
                            diffs[n][realIndex][i] = (diffs[n][realIndex-1][i] + 1/48)
                        realSeriesCopy.remove(obs)
                        if (localMin==obs or localMax==obs):
                            localMin=min(realSeriesCopy)
                            localMax=max(realSeriesCopy)
                    else:
                        cleanSeriesPerVariable[realIndex]=obs

            minAllVarAllPat[i].append(localMin)
            maxAllVarAllPat[i].append(localMax)
                
            cleanSeriesPerPatient.append(cleanSeriesPerVariable)

        cleanSeries3DMatrix.append(np.asarray(cleanSeriesPerPatient))
        
    cleanSeries3DMatrix = torch.from_numpy(np.asarray(cleanSeries3DMatrix))
    cleanSeries3DMatrix = torch.transpose(cleanSeries3DMatrix, 1, 2)
    
    for v in range(numVariables):
        minAllVar[v] = min(minn for minn in minAllVarAllPat[v] if minn is not None)
        maxAllVar[v] = max(maxx for maxx in maxAllVarAllPat[v] if maxx is not None)
    
    #return minAllVar, maxAllVar, cleanSeries3DMatrix, masks, numPatients, numTimeSteps

    normalCleanSeries = normalize(minAllVar, maxAllVar, cleanSeries3DMatrix, masks, numPatients, numTimeSteps)
    
    return normalCleanSeries, masks, diffs   #outputs tensors

