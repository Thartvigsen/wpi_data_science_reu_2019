#This code does runs the model through testing data

import numpy as np
import torch
import sklearn
from sklearn import metrics

def testRun(model, test_loader, BATCH_SIZE, criterion, fileName4, fileName5, fileName6):

    count = 0
    aucTotal = 0
    lossTotal = 0
    loss_sTotal = 0

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
        loss, loss_s = criterion(predictions, labels)
        lossTotal+=loss/BATCH_SIZE
        loss_sTotal+=loss_s/BATCH_SIZE

    lossFinal = lossTotal/count
    loss_sFinal = loss_sTotal/count

    auc = aucTotal/count

    print('Testing AUC: {}'.format(auc))
    f4 = open(fileName4, "a+")
    f5 = open(fileName5, "a+")
    f6 = open(fileName6, "a+")
    f4.write('%f,' % lossFinal)
    f5.write('%f,' % auc)
    f6.write('%f,' %loss_sFinal)

    return auc
