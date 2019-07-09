#This code does runs the model through testing data

import numpy as np
import torch
import sklearn
from sklearn import metrics

def testRun(model, test_loader):

    count = 0
    aucTotal = 0

    for i, (time_series, labels) in enumerate(test_loader):

        predictions = model(time_series)

        count += 1

        arr = np.sum(labels.data.numpy(),axis=0)
        
        for i in range(len(arr)):
            if(arr[i]==0):
                arr[i] = 1
                labelsT = torch.transpose(labels, 0, 1)
                labelsT[i][0] = 1
                labels = torch.transpose(labelsT, 0, 1)
            elif(arr[i]==model.BATCH_SIZE):
                arr[i]=9
                labelsT = torch.transpose(labels, 0, 1)
                labelsT[i][0] = 0
                labels = torch.transpose(labels, 0, 1)
        arr = np.sum(labels.data.numpy(), axis=0)
        aucTotal += sklearn.metrics.roc_auc_score(labels, predictions.detach(), average="micro")

    auc = aucTotal/count

    print('AUC: {}'.format(auc))

    return auc
